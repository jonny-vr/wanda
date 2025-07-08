import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from lib.utils import ProxyCatcher 

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

    layers = getattr(getattr(model, "model", model), "layers",
                    getattr(getattr(model, "language_model", model), "layers", None))
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

def prepare_calibration_input(model,
                              dataloader,
                              device,
                              max_calib_len: int = 2048):
    """
    Erfasst Hidden-States vor Layer 0 für die Kalibrierung.
    Gibt zurück:
        inps  – [nsamples, seq_len, hidden]  fp16
        outs  – zeros_like(inps)
        kw    – Dict aller vom Modell benötigten kwargs
    """
    # -- use_cache sicher aus- und wieder einschalten -----------------
    orig_cache = getattr(model.config, "use_cache", None)
    if orig_cache is not None:
        model.config.use_cache = False

    seq_len  = min(max_calib_len, model.seqlen)
    nsamples = len(dataloader)
    dtype    = next(model.parameters()).dtype

    embed_dev = model.hf_device_map.get("model.embed_tokens", device)
    inps = torch.zeros(
        (nsamples, seq_len,
        getattr(model.config, "hidden_size", model.config.text_config.hidden_size)),
        dtype=dtype,
        device=embed_dev,
    )
    outs = torch.zeros_like(inps)

    # -- Proxy einhängen ----------------------------------------------
    # model agnostic
    # universell in prepare_calibration_input & Co.
    print("Top-Level:", [a for a in dir(model) if not a.startswith("_")])
    if hasattr(model, "language_model"):
        print("Language-Model attrs:", dir(model.language_model))
    if hasattr(model, "model"):
        print("Model.model attrs:", dir(model.model))
    if hasattr(model, "base_model"):
        print("Base model attrs:", dir(model.base_model))

    # in prepare_calibration_input (and prune_wanda etc.)
    layers = getattr(getattr(model, "model", model), "layers",
                    getattr(getattr(model, "language_model", model), "layers", None))

    cache  = {}
    layers[0] = ProxyCatcher(layers[0], cache)

    try:
        for idx, batch in enumerate(dataloader):
            tokens = batch[0][:, :seq_len].to(device)  # hart kürzen
            try:
                model(tokens)
            except StopIteration:
                pass
            inps[idx] = cache["inp"]
            if idx + 1 == nsamples:
                break
    finally:
        layers[0] = layers[0].module  # Original-Layer zurücksetzen

    # -- cache zurücksetzen ------------------------------------------
    if orig_cache is not None:
        model.config.use_cache = orig_cache

    cache_kwargs = {k: v for k, v in cache.items() if k != "inp"}
    return inps, outs, cache_kwargs



def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = getattr(getattr(model, "model", model), "layers",
                getattr(getattr(model, "language_model", model), "layers", None))

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

# ---- safe helper -------------------------------------------------
def get_use_cache(cfg):
    return getattr(cfg, "use_cache", None)

def set_use_cache(cfg, value):
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = value
# ------------------------------------------------------------------


def prune_wanda(
        args,
        model,
        tokenizer,
        device=torch.device("cuda:0"),
        prune_n: int = 0,
        prune_m: int = 0,
):
    use_cache_orig = get_use_cache(model.config)
    set_use_cache(model.config, False)

    # ---------- 1. Calibration -------------------------------------------------
    print("loading calibration data …")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, kw = prepare_calibration_input(model, dataloader, device)
        
    inps, outs, kw = prepare_calibration_input(model, dataloader, device)

    ids = kw.get("position_ids", None)
    if ids is not None \
    and hasattr(model, "get_position_embeddings") \
    and hasattr(model.language_model, "rotary_emb_local"):

        # global
        pos_table = model.get_position_embeddings().to(device)  # [max_pos,hidden]
        ids1      = ids[:1].to(device)                         # [1, L]
        kw["position_embeddings_global"] = pos_table[ids1]     # [1, L, hidden]

        # local
        seq_len = ids1.size(-1)
        # falls rotary_emb_local schon (1,H,L,2L) liefert:
        peg_l = model.language_model.rotary_emb_local(seq_len, device=device)
        # andernfalls:
        # cos_l, sin_l = model.language_model.rotary_emb_local(seq_len, device=device)
        # peg_l        = torch.cat([cos_l.unsqueeze(0), sin_l.unsqueeze(0)], dim=-1)
        kw["position_embeddings_local"] = peg_l

    # ---------- 2. Layer-weises Pruning ---------------------------------------
    layers = getattr(getattr(model, "model", model), "layers",
                getattr(getattr(model, "language_model", model), "layers", None))
    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        # ggf. anderes Device bei Multi-GPU
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            for k, v in kw.items():
                kw[k] = v.to(dev) if torch.is_tensor(v) else v

        # --- Aktivierungen einsammeln
        wrapped = {n: WrappedGPT(m) for n, m in subset.items()}

        def add_batch(name):
            def _hook(_, inp, out):
                wrapped[name].add_batch(inp[0].data, out.data)
            return _hook

        handles = [m.register_forward_hook(add_batch(n))
                   for n, m in subset.items()]

        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **kw)[0]

        for h in handles:
            h.remove()

        # --- WANDA-Scores & Masken
        for name, mod in subset.items():
            print(f"pruning layer {i} – {name}")
            score = (torch.abs(mod.weight.data) *
                     torch.sqrt(wrapped[name].scaler_row.view(1, -1)))
            mask = torch.zeros_like(score, dtype=torch.bool)

            if prune_n:                                   # strukt. N:M
                for col in range(0, score.size(1), prune_m):
                    blk = score[:, col:col + prune_m]
                    idx = torch.topk(blk, prune_n, dim=1, largest=False).indices
                    mask.scatter_(1, col + idx, True)
            else:                                         # unstrukturiert
                sort_res = torch.sort(score, dim=-1, stable=True)
                if args.use_variant:                      # Wanda-Variant
                    tmp  = torch.cumsum(sort_res.values, dim=1)
                    tot  = score.sum(dim=1)
                    alpha_lo, alpha_hi = 0.0, 0.8
                    alpha = 0.4
                    while True:
                        mask, cur = return_given_alpha(alpha, sort_res,
                                                       score, tmp, tot)
                        if abs(cur - args.sparsity_ratio) < 1e-3 or \
                           alpha_hi - alpha_lo < 1e-3:
                            break
                        if cur > args.sparsity_ratio:
                            alpha_hi = alpha
                        else:
                            alpha_lo = alpha
                        alpha = 0.5 * (alpha_lo + alpha_hi)
                    print(f"alpha={alpha:.3f}, sparsity={cur:.4f}")
                else:
                    k = int(score.size(1) * args.sparsity_ratio)
                    idx = sort_res.indices[:, :k]
                    mask.scatter_(1, idx, True)

            mod.weight.data[mask] = 0.0

        # --- Hidden-States für nächste Schicht
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **kw)[0]
        inps, outs = outs, inps
        torch.cuda.empty_cache()

    # … nach dem Pruning am Ende:
    if use_cache_orig is not None:
        set_use_cache(model.config, use_cache_orig)
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = getattr(getattr(model, "model", model), "layers",
                    getattr(getattr(model, "language_model", model), "layers", None))

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
            gpts[name] = SparseGPT(subset[name])

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

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = getattr(getattr(model, "model", model), "layers",
                    getattr(getattr(model, "language_model", model), "layers", None))

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