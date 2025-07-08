# Import necessary modules
import time
import torch
import torch.nn as nn
from .data import get_loaders 
from collections import defaultdict
import fnmatch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from model_configs import get_model_cfg
import math, re, torch
from datasets import load_dataset

def _try_batch(model, ids, bs, first_dev, need_attn_mask=False):
    """Return True if forward pass fits in memory for given batch size."""
    bs = min(bs, ids.size(0))
    inp = ids[:bs].to(first_dev)
    tgt = inp.clone(); tgt[:, 0] = -100
    amask = torch.ones_like(inp).to(first_dev) if need_attn_mask else None
    try:
        with torch.no_grad():
            _ = model(inp, attention_mask=amask, labels=tgt, use_cache=False)
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False
        raise


def compute_wikitext_ppl(model,
                         tokenizer,
                         device,
                         batch_size: str | int = "auto"):
    """
    Perplexity on WikiText-2-raw-v1 test split using fixed-length blocks.
    • context length = cfg["max_len"] (falls back to 2048)
    • `use_cache=False` to minimise VRAM
    • optional automatic batch-size tuning
    Returns: ppl (float)
    """
    model.eval()
    cfg = get_model_cfg(model.config._name_or_path.split("/")[-1])
    block_size = cfg.get("max_len", 2048)

    # pick "first" device for forward calls
    if getattr(model, "hf_device_map", None):
        first_dev = next(iter(model.hf_device_map.values()))
    else:
        first_dev = device

    # need_attn_mask flag for Llama-4 Scout variants
    need_attn_mask = bool(re.match(r"^Llama-4-Scout", model.config._name_or_path))

    # ------------------- dataset ------------------- #
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text,
                       add_special_tokens=False,
                       return_tensors="pt").input_ids[0]

    if tokenizer.bos_token_id is not None:
        tokens = torch.cat([torch.tensor([tokenizer.bos_token_id]), tokens])

    nsamples = tokens.numel() // block_size
    if nsamples == 0:
        raise ValueError(f"Not enough tokens for block_size={block_size}")

    ids = tokens[: nsamples * block_size].view(nsamples, block_size)

    # --------------- batch-size autotune ------------ #
    if batch_size == "auto":
        MAX_BS = 16
        best, bs = 1, 1
        while bs <= MAX_BS:
            if _try_batch(model, ids, bs, first_dev, need_attn_mask):
                best, bs = bs, bs * 2
            else:
                bs //= 2; break
        batch_size = best
    else:
        batch_size = int(batch_size)

    # ------------------- main loop ------------------ #
    nll, tok_cnt = 0.0, 0
    i, bs_cur = 0, batch_size
    while i < nsamples:
        j = min(i + bs_cur, nsamples)
        inp = ids[i:j].to(first_dev)
        tgt = inp.clone(); tgt[:, 0] = -100
        amask = torch.ones_like(inp).to(first_dev) if need_attn_mask else None

        try:
            with torch.no_grad():
                loss = model(inp,
                             attention_mask=amask,
                             labels=tgt,
                             use_cache=False).loss.item()
            toks = tgt.ne(-100).sum().item()
            nll  += loss * toks
            tok_cnt += toks
            i += bs_cur
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs_cur > 1:
                torch.cuda.empty_cache()
                bs_cur //= 2
                print(f"⚠️  OOM – reduce batch_size to {bs_cur}, retry")
            else:
                raise

    return math.exp(nll / tok_cnt)



# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    
    
    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer, 
        add_special_tokens=add_special_tokens
    )

    return results 