# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('allenai/c4', 'en', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', trust_remote_code=True)
    valdata = load_dataset('allenai/c4', 'en', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', trust_remote_code=True)
    
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Load GSM8K dataset for mathematical reasoning
def get_gsm8k(nsamples, seed, seqlen, tokenizer):
    dataset = load_dataset('gsm8k', 'main', split='train')
    
    random.seed(seed)
    trainloader = []
    
    for _ in range(nsamples):
        # Sample a random problem
        idx = random.randint(0, len(dataset) - 1)
        problem = dataset[idx]
        
        # Format as "Question: ... Answer: ..."
        text = f"Question: {problem['question']}\nAnswer: {problem['answer']}"
        
        # Tokenize
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=seqlen)
        
        if enc.input_ids.shape[1] < seqlen:
            # Pad with additional problems if needed
            while enc.input_ids.shape[1] < seqlen:
                idx = random.randint(0, len(dataset) - 1)
                problem = dataset[idx]
                additional_text = f"\n\nQuestion: {problem['question']}\nAnswer: {problem['answer']}"
                new_enc = tokenizer(text + additional_text, return_tensors='pt', truncation=True, max_length=seqlen)
                if new_enc.input_ids.shape[1] <= seqlen:
                    text += additional_text
                    enc = new_enc
                else:
                    break
        
        inp = enc.input_ids[:, :seqlen]
        if inp.shape[1] < seqlen:
            # Pad to seqlen
            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), value=tokenizer.pad_token_id)
        
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    # For validation, we'll use a subset of test data
    testdata = load_dataset('gsm8k', 'main', split='test')
    val_texts = []
    for i in range(min(100, len(testdata))):
        val_texts.append(f"Question: {testdata[i]['question']}\nAnswer: {testdata[i]['answer']}")
    
    valenc = tokenizer('\n\n'.join(val_texts), return_tensors='pt', truncation=True, max_length=256*seqlen)
    return trainloader, valenc

# Load ARC dataset for science reasoning
def get_arc(nsamples, seed, seqlen, tokenizer):
    # Using ARC-Challenge for harder reasoning questions
    dataset = load_dataset('ai2_arc', 'ARC-Challenge', split='train')
    
    random.seed(seed)
    trainloader = []
    
    for _ in range(nsamples):
        samples_text = ""
        
        while len(tokenizer(samples_text, return_tensors='pt').input_ids[0]) < seqlen:
            idx = random.randint(0, len(dataset) - 1)
            item = dataset[idx]
            
            # Format question with choices
            question_text = f"Question: {item['question']}\n"
            for i, choice in enumerate(item['choices']['text']):
                question_text += f"{item['choices']['label'][i]}. {choice}\n"
            question_text += f"Answer: {item['answerKey']}\n\n"
            
            if len(tokenizer(samples_text + question_text, return_tensors='pt').input_ids[0]) <= seqlen:
                samples_text += question_text
            else:
                break
        
        enc = tokenizer(samples_text, return_tensors='pt', truncation=True, max_length=seqlen)
        inp = enc.input_ids[:, :seqlen]
        
        if inp.shape[1] < seqlen:
            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), value=tokenizer.pad_token_id)
        
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    # Validation data
    testdata = load_dataset('ai2_arc', 'ARC-Challenge', split='validation')
    val_texts = []
    for i in range(min(50, len(testdata))):
        item = testdata[i]
        question_text = f"Question: {item['question']}\n"
        for j, choice in enumerate(item['choices']['text']):
            question_text += f"{item['choices']['label'][j]}. {choice}\n"
        question_text += f"Answer: {item['answerKey']}"
        val_texts.append(question_text)
    
    valenc = tokenizer('\n\n'.join(val_texts), return_tensors='pt', truncation=True, max_length=256*seqlen)
    return trainloader, valenc

# Load CommonsenseQA dataset
def get_commonsenseqa(nsamples, seed, seqlen, tokenizer):
    dataset = load_dataset('commonsense_qa', split='train')
    
    random.seed(seed)
    trainloader = []
    
    for _ in range(nsamples):
        samples_text = ""
        
        while len(tokenizer(samples_text, return_tensors='pt').input_ids[0]) < seqlen:
            idx = random.randint(0, len(dataset) - 1)
            item = dataset[idx]
            
            # Format question with choices
            question_text = f"Question: {item['question']}\n"
            for choice in item['choices']['text']:
                question_text += f"- {choice}\n"
            question_text += f"Answer: {item['answerKey']}\n\n"
            
            if len(tokenizer(samples_text + question_text, return_tensors='pt').input_ids[0]) <= seqlen:
                samples_text += question_text
            else:
                break
        
        enc = tokenizer(samples_text, return_tensors='pt', truncation=True, max_length=seqlen)
        inp = enc.input_ids[:, :seqlen]
        
        if inp.shape[1] < seqlen:
            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), value=tokenizer.pad_token_id)
        
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    # Validation data
    testdata = load_dataset('commonsense_qa', split='validation')
    val_texts = []
    for i in range(min(50, len(testdata))):
        item = testdata[i]
        question_text = f"Question: {item['question']}\n"
        for choice in item['choices']['text']:
            question_text += f"- {choice}\n"
        question_text += f"Answer: {item['answerKey']}"
        val_texts.append(question_text)
    
    valenc = tokenizer('\n\n'.join(val_texts), return_tensors='pt', truncation=True, max_length=256*seqlen)
    return trainloader, valenc

# Mixed reasoning dataset - combines multiple reasoning datasets
def get_mixed_reasoning(nsamples, seed, seqlen, tokenizer):
    random.seed(seed)
    
    # Load all datasets
    gsm8k = load_dataset('gsm8k', 'main', split='train')
    print("Loaded GSM8K dataset")
    arc = load_dataset('ai2_arc', 'ARC-Challenge', split='train')
    print("Loaded ARC dataset")
    csqa = load_dataset('commonsense_qa', split='train')
    print("Loaded CommonsenseQA dataset")
    
    trainloader = []
    datasets = ['gsm8k', 'arc', 'csqa']
    
    for _ in range(nsamples):
        samples_text = ""
        
        while len(tokenizer(samples_text, return_tensors='pt').input_ids[0]) < seqlen:
            # Randomly choose a dataset
            dataset_choice = random.choice(datasets)
            
            if dataset_choice == 'gsm8k':
                idx = random.randint(0, len(gsm8k) - 1)
                problem = gsm8k[idx]
                text = f"Math Problem: {problem['question']}\nSolution: {problem['answer']}\n\n"
            
            elif dataset_choice == 'arc':
                idx = random.randint(0, len(arc) - 1)
                item = arc[idx]
                text = f"Science Question: {item['question']}\n"
                for i, choice in enumerate(item['choices']['text']):
                    text += f"{item['choices']['label'][i]}. {choice}\n"
                text += f"Answer: {item['answerKey']}\n\n"
            
            else:  # csqa
                idx = random.randint(0, len(csqa) - 1)
                item = csqa[idx]
                text = f"Commonsense Question: {item['question']}\n"
                for choice in item['choices']['text']:
                    text += f"- {choice}\n"
                text += f"Answer: {item['answerKey']}\n\n"
            
            if len(tokenizer(samples_text + text, return_tensors='pt').input_ids[0]) <= seqlen:
                samples_text += text
            else:
                break
        
        enc = tokenizer(samples_text, return_tensors='pt', truncation=True, max_length=seqlen)
        inp = enc.input_ids[:, :seqlen]
        
        if inp.shape[1] < seqlen:
            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), value=tokenizer.pad_token_id)
        
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    # Create mixed validation set
    val_texts = []
    
    # Get some examples from each dataset's validation split
    gsm8k_val = load_dataset('gsm8k', 'main', split='test')
    arc_val = load_dataset('ai2_arc', 'ARC-Challenge', split='validation')
    csqa_val = load_dataset('commonsense_qa', split='validation')
    
    for i in range(min(20, len(gsm8k_val))):
        val_texts.append(f"Math Problem: {gsm8k_val[i]['question']}\nSolution: {gsm8k_val[i]['answer']}")
    
    for i in range(min(20, len(arc_val))):
        item = arc_val[i]
        text = f"Science Question: {item['question']}\n"
        for j, choice in enumerate(item['choices']['text']):
            text += f"{item['choices']['label'][j]}. {choice}\n"
        text += f"Answer: {item['answerKey']}"
        val_texts.append(text)
    
    for i in range(min(20, len(csqa_val))):
        item = csqa_val[i]
        text = f"Commonsense Question: {item['question']}\n"
        for choice in item['choices']['text']:
            text += f"- {choice}\n"
        text += f"Answer: {item['answerKey']}"
        val_texts.append(text)
    
    valenc = tokenizer('\n\n'.join(val_texts), return_tensors='pt', truncation=True, max_length=256*seqlen)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=1028, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "gsm8k" in name:
        return get_gsm8k(nsamples, seed, seqlen, tokenizer)
    elif "arc" in name:
        return get_arc(nsamples, seed, seqlen, tokenizer)
    elif "commonsenseqa" in name or "csqa" in name:
        return get_commonsenseqa(nsamples, seed, seqlen, tokenizer)
    elif "mixed_reasoning" in name or "mixed-reasoning" in name:
        return get_mixed_reasoning(nsamples, seed, seqlen, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {name}")