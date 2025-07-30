import time 
import inspect
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import torch.nn.utils.prune as prune
import inspect

from .ablate import AblateGPT 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device, calib_len=2048):
    """
    Collect `nsamples × calib_len` hidden-states from the first decoder block
    and return them together with the exact kwargs that were fed into the model.
    Works for Qwen-3 (needs position_embeddings) *and* Llama-family
    (needs position_ids).

    Returns
    -------
    inps :  (nsamples, calib_len, hidden)  tensor
    outs :  same shape,      initialised to zeros
    kw   :  dict with attention_mask / position_ids / position_embeddings
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # if the model was automatically sharded -> use the same GPU as the embeddings
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype   = next(iter(model.parameters())).dtype
    nsample = len(dataloader)                 # should equal args.nsamples
    inps    = torch.zeros((nsample, calib_len, model.config.hidden_size),
                          dtype=dtype, device=device)

    # shared bag where the Catcher stores activations + the first kwargs
    store = {"inps": inps,
             "i":   0,
             "kw":  {}}

    class Catcher(nn.Module):
        def __init__(self, layer, bag):
            super().__init__()
            self._orig, self._bag = layer, bag

        def forward(self, hidden_states, **kwargs):
            b = self._bag
            b["inps"][b["i"]] = hidden_states
            b["i"] += 1
            if not b["kw"]:               # save kwargs only once
                b["kw"] = kwargs
            raise ValueError              # stop forward pass here

        # delegate every unknown attribute to the wrapped layer
        def __getattr__(self, name):
            if name in {"_orig", "_bag"}:
                return super().__getattr__(name)
            return getattr(self._orig, name)

    # wrap the first decoder block
    first = model.model.layers[0]
    model.model.layers[0] = Catcher(first, store)

    for batch in dataloader:
        try:
            model(batch[0].to(device)[:, :calib_len])
        except ValueError:
            pass                           # Catcher always raises

    # unwrap
    model.model.layers[0] = first
    model.config.use_cache = use_cache

    outs = torch.zeros_like(inps)
    return inps, outs, store["kw"]


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0



# -------------------------------------------------------------------------
# Wanda-Pruning mit persistenter Maske
# -------------------------------------------------------------------------
@torch.no_grad()
def prune_wanda(
    args,
    model,
    tokenizer,
    device=torch.device("cuda:0"),
    prune_n: int = 0,
    prune_m: int = 0,
):
    """
    Wanda-Pruning, aber die Null-Gewichte werden per weight_mask dauerhaft
    fixiert, so dass sie sich bei weiterem Training nicht „wiederbeleben“.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # 0) Kalibrier-Daten vorbereiten (wie im Original)
    calib_len = 4096
    print("loading calibration data …")
    print(f"loading calibration data from {args.calib_dataset}...")
    dataloader, _ = get_loaders(
        args.calib_dataset,  # Changed from hardcoded "wikitext2"
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    with torch.no_grad():
        inps, outs, replay_kw = prepare_calibration_input(
            model, dataloader, device, calib_len
        )
    print("dataset loading complete")

    attn_mask = replay_kw.get("attention_mask")
    pos_ids   = replay_kw.get("position_ids")
    pos_emb   = replay_kw.get("position_embeddings")   # Qwen-3 / DeepSeek

    layers = model.model.layers
    # ------------------------------------------------------------------
    # 1) Schicht für Schicht: Statistiken sammeln → Maske ableiten → anwenden
    # ------------------------------------------------------------------
    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        # ggf. auf das richtige GPU-Shard umziehen (bei mp-sharding)
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attn_mask is not None:
                attn_mask = attn_mask.to(dev)
            if pos_ids is not None:
                pos_ids = pos_ids.to(dev)
            if pos_emb is not None:
                pos_emb = tuple(p.to(dev) for p in pos_emb)

        # ---------- Aktivitäts-Statistik sammeln ----------
        wrapped = {n: WrappedGPT(m) for n, m in subset.items()}

        def add_batch(name):
            def _hook(_, inp, out):
                wrapped[name].add_batch(inp[0].data, out.data)
            return _hook

        hooks = [m.register_forward_hook(add_batch(n)) for n, m in subset.items()]
        for j in range(args.nsamples):
            sig = inspect.signature(layer.forward)
            kw  = {}
            if attn_mask is not None: kw["attention_mask"] = attn_mask
            if pos_ids   is not None: kw["position_ids"]   = pos_ids
            if pos_emb   is not None and "position_embeddings" in sig.parameters:
                kw["position_embeddings"] = pos_emb
            outs[j] = layer(inps[j].unsqueeze(0), **kw)[0]
        for h in hooks: h.remove()

        # -------------------- Wanda-Masken ableiten --------------------
        for name, mod in subset.items():
            print(f"Layer {i} – {name}: compute Wanda mask")
            W = mod.weight.data
            metric = torch.abs(W) * torch.sqrt(
                wrapped[name].scaler_row.reshape(1, -1)
            )

            if prune_n:                                   # strukturiert N:M
                mask_bool = torch.zeros_like(W, dtype=torch.bool)
                for col in range(0, metric.shape[1], prune_m):
                    blk   = metric[:, col:col + prune_m]
                    topk  = torch.topk(blk, prune_n, dim=1, largest=False).indices
                    mask_bool.scatter_(1, col + topk, True)
            else:                                         # unstrukturiert
                k = int(metric.numel() * args.sparsity_ratio)
                thresh = torch.topk(metric.view(-1), k, largest=False).values.max()
                mask_bool = metric <= thresh              # True == prune

            # -----------------------------------------------------------
            # Persistente Maske via torch.prune
            # -----------------------------------------------------------
            # torch-Mask soll 1 = KEEP, 0 = PRUNE ⇒ invertieren
            torch_mask = (~mask_bool).to(W.device)
            prune.custom_from_mask(mod, name="weight", mask=torch_mask)
            # weight = weight_orig * weight_mask  (forward-zeit)

        # ----------- Puffer tauschen (für nächste Schicht) --------------
        for j in range(args.nsamples):
            sig = inspect.signature(layer.forward)
            kw  = {}
            if attn_mask is not None: kw["attention_mask"] = attn_mask
            if pos_ids   is not None: kw["position_ids"]   = pos_ids
            if pos_emb   is not None and "position_embeddings" in sig.parameters:
                kw["position_embeddings"] = pos_emb
            outs[j] = layer(inps[j].unsqueeze(0), **kw)[0]
        inps, outs = outs, inps   # flip buffers

    # ------------------------------------------------------------------
    # 2) Aufräumen
    # ------------------------------------------------------------------
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Wanda pruning finished – masks are now persistent.")


@torch.no_grad()
@torch.no_grad()
def prune_sparsegpt(
    args,
    model,
    tokenizer,
    dev=torch.device("cuda:0"),
    prune_n: int = 0,
    prune_m: int = 0,
):
    """
    SparseGPT-Pruning mit Qwen/DeepSeek-Kompatibilität:
    * nutzt prepare_calibration_input → verarbeitet attention_mask, position_ids,
      UND position_embeddings (falls vorhanden)
    * reicht position_embeddings nur weiter, wenn das jeweilige Layer sie erwartet
    * funktioniert weiterhin mit (optionaler) Geräte-Shard-Verteilung
    """
    print("Starting SparseGPT pruning …")

    # ----------------------------------------------------------
    # 1) Kalibrierdaten laden und Aktivierungen einsammeln
    # ----------------------------------------------------------
    print(f"loading calibration data from {args.calib_dataset}...")
    dataloader, _ = get_loaders(
        args.calib_dataset,  # Changed from hardcoded "wikitext2"
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # korrekte GPU für die Embeddings (wichtig bei 30B/65B-Sharding)
    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    # etwas kürzeres Kalibrier-Fenster zur Beschleunigung (wie Wanda)
    calib_len = 4096
    with torch.no_grad():
        inps, outs, replay_kw = prepare_calibration_input(
            model, dataloader, dev, calib_len
        )

    # ausgepackte Keyword-Argumente (können None sein)
    attn_mask = replay_kw.get("attention_mask")
    pos_ids   = replay_kw.get("position_ids")
    pos_emb   = replay_kw.get("position_embeddings")   # nur bei Qwen-3/DeepSeek

    layers = model.model.layers

    # ----------------------------------------------------------
    # 2) Schicht für Schicht: Statistiken sammeln → prunen
    # ----------------------------------------------------------
    for i, layer in enumerate(layers):
        # ggf. auf das GPU-Shard dieser Schicht umziehen
        if f"model.layers.{i}" in model.hf_device_map:
            dev_layer = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev_layer)
            outs = outs.to(dev_layer)
            if attn_mask is not None:
                attn_mask = attn_mask.to(dev_layer)
            if pos_ids is not None:
                pos_ids = pos_ids.to(dev_layer)
            if pos_emb is not None:
                pos_emb = tuple(p.to(dev_layer) for p in pos_emb)

        subset = find_layers(layer)
        gpts   = {n: SparseGPT(m) for n, m in subset.items()}

        # ---------------- Aktivitäten sammeln -----------------
        def add_batch(name):
            def _hook(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return _hook

        handles = [
            m.register_forward_hook(add_batch(n)) for n, m in subset.items()
        ]

        for j in range(args.nsamples):
            sig = inspect.signature(layer.forward)
            call_kwargs = {}
            if attn_mask is not None:
                call_kwargs["attention_mask"] = attn_mask
            if pos_ids is not None:
                call_kwargs["position_ids"] = pos_ids
            # nur mitsenden, wenn das Layer position_embeddings erwartet
            if pos_emb is not None and "position_embeddings" in sig.parameters:
                call_kwargs["position_embeddings"] = pos_emb
                
            outs[j] = layer(inps[j].unsqueeze(0), **call_kwargs)[0]

        for h in handles:
            h.remove()

        # ----------------------- Pruning -----------------------
        for name, gpt in gpts.items():
            print(f"Layer {i} – {name}: pruning …")
            gpt.fasterprune(
                args.sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpt.free()

        # ------------- Outputs neu berechnen -------------------
        for j in range(args.nsamples):
            sig = inspect.signature(layer.forward)
            call_kwargs = {}
            if attn_mask is not None:
                call_kwargs["attention_mask"] = attn_mask
            if pos_ids is not None:
                call_kwargs["position_ids"] = pos_ids
            # nur mitsenden, wenn das Layer position_embeddings erwartet
            if pos_emb is not None and "position_embeddings" in sig.parameters:
                call_kwargs["position_embeddings"] = pos_emb
                
            outs[j] = layer(inps[j].unsqueeze(0), **call_kwargs)[0]

        torch.cuda.empty_cache()
        inps, outs = outs, inps  # Buffer tauschen

    # ----------------------------------------------------------
    # 3) Aufräumen
    # ----------------------------------------------------------
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("SparseGPT pruning finished.")



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("wikitext2",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()