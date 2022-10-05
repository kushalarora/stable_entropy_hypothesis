import hashlib
import os
from shutil import register_unpack_format
from typing import List, Union

import copy
import json
import pandas as pd
import math

import numpy as np
import torch.nn.functional as F
from termcolor import colored
import torch

from entropy_aware_search.hf_utils import DataArguments, ModelArguments, get_tokenizer, get_model

COLUMNS = ['index', 'token', 'is_rep', 'is_lrep', 
            'is_crep', 'entropy', 'entropy_ma', 'dent', 
            'ddent', 'dddent', 'probs', 'rank']

def entropy_from_scores(scores):
    logits = F.log_softmax(scores, dim=-1)
    entropy = (-1 * logits.exp() * logits).sum(-1)
    return entropy

def moving_average(arr, w=4):
    left = w 
    right = 0
    return np.array([np.mean(arr[max(i-left,0):i+right]) for i in range(1, len(arr)+1)])

def running_average(arr):
   ma_arr = []
   n = 1
   sum_x = 0
   for x in arr:
       sum_x += x
       ma_arr.append(sum_x/n)
       n += 1
   return ma_arr

def compute_ngram_repeats(context: Union[str, List], model_text: Union[str, List], n=3, splitted=False):
    cgrams = {}
    # compute N grams of the context
    
    if not splitted:
        context = context.split(' ')
        model_text = model_text.split(' ')

    for i in range(n, len(context) + 1):
        ngram = ' '.join(context[i - n : i])
        cgrams[ngram] = True
    # compute N grams of the model response
    creps = 0
    lreps = 0
    repetition_idxs = [0] * (n - 1)
    lreps_idxs = [0] * (n - 1)
    creps_idxs = [0] * (n - 1)
    
    lgrams = {}

    for i in range(n, len(model_text) + 1):
        ngram = ' '.join(model_text[i - n : i])
        if ngram in cgrams:
            creps = creps + 1
            repetition_idxs.append(1) 
            creps_idxs.append(1)
            lreps_idxs.append(0) 
        elif ngram in lgrams:
            lreps = lreps + 1
            repetition_idxs.append(1)
            lreps_idxs.append(1) 
            creps_idxs.append(0)
        else:
            repetition_idxs.append(0) 
            creps_idxs.append(0)
            lreps_idxs.append(0)
        lgrams[ngram] = True
    
    for i in range(len(repetition_idxs) - n):
        if repetition_idxs[i+n-1] == 1:
            for j in range(n):
                repetition_idxs[i+j] = 1

    for i in range(len(creps_idxs) - n):
        if creps_idxs[i+n-1] == 1:
            for j in range(n):
                creps_idxs[i+j] = 1

    for i in range(len(lreps_idxs) - n):
        if lreps_idxs[i+n-1] == 1:
            for j in range(n):
                lreps_idxs[i+j] = 1
    return creps + lreps, creps, lreps, repetition_idxs, creps_idxs, lreps_idxs

def repeat_analysis(context, model_text):    
    (num_3_gram_repeat, creps_num_3_gram_repeat, 
        lreps_num_3_gram_repeat, rep_idxs, crep_idxs, lrep_idxs) = \
                            compute_ngram_repeats(context, model_text)

    num_3_gram_repeat_length_normalized = int(num_3_gram_repeat/len(model_text.split(" ")) * 52)
    has_3_gram_repeat = num_3_gram_repeat > 0
    has_3_gram_label_repeat = lreps_num_3_gram_repeat > 0

    return {
        'num_3_gram_repeat': num_3_gram_repeat,
        'has_3_gram_repeat': has_3_gram_repeat,
        'lreps_num_3_gram_repeat': lreps_num_3_gram_repeat,
        'creps_num_3_gram_repeat': creps_num_3_gram_repeat,
        'has_3_gram_label_repeat': has_3_gram_label_repeat,
        "num_3_gram_repeat_length_normalized": num_3_gram_repeat_length_normalized,
        "rep_idxs": rep_idxs,
        "crep_idxs": crep_idxs,
        "lrep_idxs": lrep_idxs,
     }


def predict(model, tokenizer, context: str, model_text: str, width=1, max_len=128, is_seq2seq=False, max_source_len=None):
    """
    Use model to predict given the context.

    :param context:
        The input context to classify.

    :return output:
        return output from model act.
    """
    device = next(model.parameters()).device
    if is_seq2seq:
        assert max_source_len is not None

        batch = tokenizer(context, 
                    max_length=max_source_len,
                    padding=False, truncation=True,
                    return_tensors='pt')
        batch = batch.to(device)

        # Tokenize targets with the `text_target` keyword argument
        tokenized_model_text = tokenizer(text_target=model_text, 
                            max_length=max_len,
                            padding=False, truncation=True, 
                            return_tensors='pt')
        tokenized_model_text = tokenized_model_text.to(device)

        batch["labels"] = tokenized_model_text["input_ids"]
        outputs = model(**batch)
        model_scores = outputs[1]

        tokenized_context = batch
        next_tokens = tokenized_model_text['input_ids']

    else:
        tokenized_context = tokenizer(context, return_tensors="pt")
        prompt_len = tokenized_context['input_ids'].shape[1]

        batch = tokenizer(context, model_text, max_length=1024, return_tensors="pt")
        
        batch = batch.to(device)
        tokenized_context = tokenized_context.to(device)
        outputs = model(**batch)
        
        model_scores = outputs[0][:, prompt_len:-1]
        tokenized_model_text = tokenizer(model_text, max_length=1024-prompt_len, return_tensors="pt")
        next_tokens = batch['input_ids'][:, prompt_len+1:]

    entropy = entropy_from_scores(model_scores)
    

    model_probs = F.softmax(model_scores, dim=-1)
    next_token_probs = model_probs\
                        .gather(-1, next_tokens.unsqueeze(-1))\
                        .squeeze(-1)
    
    ranks = []
    _, model_probs_sorted_idxs = torch.sort(model_probs, 
                                    dim=-1, descending=True)
    
    for i in range(len(next_tokens[0])):
        ith_model_probs = model_probs_sorted_idxs[0, i].cpu().numpy() 
        ranks.append(int(np.where(ith_model_probs == int(next_tokens[0,i]))[0]))
        
    label_toks = tokenizer.convert_ids_to_tokens(tokenized_model_text['input_ids'][0])
    context_toks = tokenizer.convert_ids_to_tokens(tokenized_context['input_ids'][0])

    entropy = entropy[0].detach().cpu().numpy()
    entropy_ma = moving_average(entropy, w=width)
    # entropy_ma = running_average(entropy)

    dent_dt = copy.copy(entropy_ma)
    dent_dt[width+1:] = (dent_dt[width+1:] - dent_dt[:-(width+1)])
    dent_dt[:width+1] = 0
    
    ddent_dtdt = copy.copy(dent_dt)
    ddent_dtdt[1:] = dent_dt[1:] - dent_dt[:-1]
    ddent_dtdt[:2] = 0

    dddent_dtdtdt = copy.copy(ddent_dtdt)
    dddent_dtdtdt[1:] = ddent_dtdt[1:]  - ddent_dtdt[:-1]
    dddent_dtdtdt[:3] = 0

    (num_3_gram_repeat, creps_num_3_gram_repeat, 
        lreps_num_3_gram_repeat, rep_idxs, crep_idxs, lrep_idxs) = \
            compute_ngram_repeats(context_toks, label_toks, splitted=True)

    has_3_gram_repeat = num_3_gram_repeat > 0
    has_3_gram_label_repeat = lreps_num_3_gram_repeat > 0
    
    return {
        'tokens': label_toks,
        'context_tokens': context_toks,
        "probs": next_token_probs[0].cpu().tolist(),
        "ranks": ranks,
        "entropy": entropy,
        "entropy_ma": entropy_ma,
        "dentropy":dent_dt,
        "ddentropy": ddent_dtdt,
        "dddentropy": dddent_dtdtdt,
        "num_3_gram_repeat": num_3_gram_repeat,
        "has_3_gram_repeat": has_3_gram_repeat,
        "rep_idxs": rep_idxs,
        "has_3_gram_label_repeat": has_3_gram_label_repeat,
        "lrep_idxs": lrep_idxs,
        "crep_idxs": crep_idxs,
    }


def process_datapoint(model, tokenizer, datapoint, width=1, max_len=128, is_seq2seq=False, max_source_len=None):
    "Returns list of tuples, with item at idx t, (token_t, label[token_t])."
    outputs = []
    model_text = datapoint['model_text'].strip()
    context = datapoint['context'].strip()
    model_act = predict(model=model, tokenizer=tokenizer, 
                    context=context, model_text=model_text, 
                    width=width, max_len=max_len, 
                    is_seq2seq=is_seq2seq, max_source_len=max_source_len)

    # target_act = predict(model=model, context=context, model_text=eval_label, is_edc=True)

    for idx, (token, is_rep, is_lrep, is_crep, entropy, entropy_ma, 
              dentropy, ddentropy, dddentropy, prob, rank) in enumerate(zip(model_act['tokens'],
                    model_act['rep_idxs'], model_act['lrep_idxs'], model_act['crep_idxs'],
                    model_act['entropy'], model_act['entropy_ma'], model_act['dentropy'],
                    model_act['ddentropy'], model_act['dddentropy'], model_act['probs'], model_act['ranks'])):
        if token == '__end__':
            break
        outputs.append((idx, token, is_rep, is_lrep, is_crep, entropy, 
                        entropy_ma, dentropy, ddentropy,  dddentropy, prob, rank))
    return pd.DataFrame(outputs, columns=COLUMNS)


def compute_average_across_sequences(dataframe, model, tokenizer, 
        column_prefix, max_len=128, num_seq=500, to_be_averaged='entropy',
         width=1, cache=True, overwrite=False, is_seq2seq=False, 
         max_source_len=None,):

    cache_dirname = "/home/mila/a/arorakus/wdir/entropy_aware_search/data/cahced/"
    dataframe_hash = str(hashlib.md5(dataframe.to_string().encode()).hexdigest())
    model_hash = str(hashlib.md5(model.config._name_or_path.encode()).hexdigest())
    cached_filename_prefix = os.path.join(cache_dirname, 
        f"{dataframe_hash}-{model_hash}-"+
        f"{column_prefix}-{str(max_len)}-{str(num_seq)}-{to_be_averaged}-{str(width)}")


    if is_seq2seq and max_source_len is not None:
        cached_filename_prefix += f'-{str(max_source_len)}'

    cached_ndarray_filename = f"{cached_filename_prefix}-ndarray.npy"
    cached_pd_filename = f"{cached_filename_prefix}-avgs.csv"

    found = False
    if cache and \
        os.path.exists(cached_ndarray_filename) and \
            os.path.join(cached_pd_filename) and \
                not overwrite:
        print(f"Loading from:")
        print(f"\t{cached_ndarray_filename}, and")
        print(f"\t{cached_pd_filename}")

        values = np.load(cached_ndarray_filename)
        avgs_pd = pd.read_csv(cached_pd_filename)

        found = True

    if not found:
        counts = [0] * max_len
        cumls = [0.] * max_len
        values = np.zeros((num_seq, max_len))
        values.fill(-1)

        for j, (_,datapoint) in enumerate(dataframe.sample(num_seq).iterrows()):
            labeled_dataframe = process_datapoint(model=model, 
                                    tokenizer=tokenizer, datapoint=datapoint,
                                    width=width, max_len=max_len, 
                                    is_seq2seq=is_seq2seq,
                                    max_source_len=max_source_len)
            
            for i,ent in labeled_dataframe[to_be_averaged].iteritems():
                if i >= max_len:
                    break
                
                values[j, i] = ent
                cumls[i] += ent
                counts[i] += 1

        avgs = [cuml_ent/count if count > 0 else 0
                    for (cuml_ent, count) in zip(cumls, counts)]
        avgs_pd = pd.DataFrame(avgs,  columns=[column_prefix + '_' + to_be_averaged])

        if cache and \
            not found or overwrite:

            np.save(cached_ndarray_filename, values)
            avgs_pd.to_csv(cached_pd_filename)
            print(f"Saving to:")
            print(f"\t{cached_ndarray_filename}, and")
            print(f"\t{cached_pd_filename}")

    masked_avgs = np.ma.masked_array(values, mask=(values < 0))
    return avgs_pd, masked_avgs


def print_sample(sample):
    context = sample.context.item()
    model_text = sample.model_text.item()
    _, _, _, rep_idxs, _, _ = compute_ngram_repeats(context, model_text)
    # print(context)
    print(print_with_colors(model_text, rep_idxs))

    print(f"Num 3 gram repeats: {sample.num_3_gram_repeat.item()}")
    print(f"Num 3 gram repeats normalized: {sample.num_3_gram_repeat_length_normalized.item()}")


def print_with_colors(text, repeat_indices):
    colorized_tokens = []
    tokenized_text = text.split(" ")
    
    is_repeat_indices = copy.copy(repeat_indices)
    for (token, is_repeat) in zip(tokenized_text, is_repeat_indices):
       
        if is_repeat:
            colorized_token = colored(token, "red")
        else:
            colorized_token = token

        colorized_tokens.append(colorized_token)

    return " ".join(colorized_tokens)



def compute_entropy_voilations(prefixes, human_texts, generated_texts, model_name = 'gpt2-xl',
         max_len=128, num_seq=1000, width=5, is_seq2seq=False, max_source_len=-1, std_margin=1.5):

    # get model and tokenizer.
    model_args = ModelArguments(
        model_name_or_path=model_name,   
    )
    gpt2_model = get_model(model_args)
    gpt2_model.to('cuda')
    tokenizer = get_tokenizer(model_args)
    tokenizer.pad_token = tokenizer.eos_token
    gpt2_model = gpt2_model.to('cuda')

    # Compute human avg smoothened entropy.
    human_dataframe = pd.DataFrame({
        'context': prefixes,
        'model_text': human_texts
    })
    _, human_ma_entropies = compute_average_across_sequences(human_dataframe, gpt2_model, tokenizer,                            
                                column_prefix='human_generated', width=width,  max_len=max_len, 
                                to_be_averaged='entropy_ma', num_seq=num_seq, cache=True)

    human_entropy_mean = np.ma.mean(human_ma_entropies, axis=0)
    human_entropy_std = np.ma.std(human_ma_entropies, axis=0)

    generated_dataframe = pd.DataFrame({
        'context': prefixes,
        'model_text': generated_texts
    })

    entropy_violations = 0
    count = 0
    upper_bound_violations = 0
    lower_bound_violations = 0
    entropy_violation_arr = [0.] * max_len
    upper_bound_violation_arr = [0.] * max_len
    lower_bound_violation_arr = [0.] * max_len
    count_arr = [0.] * max_len
    for j, (_,datapoint) in enumerate(generated_dataframe.sample(num_seq).iterrows()):
        labeled_data = predict(model=gpt2_model, 
                                tokenizer=tokenizer, 
                                context=datapoint.context,
                                model_text=datapoint.model_text,
                                width=width, max_len=max_len, 
                                is_seq2seq=is_seq2seq,
                                max_source_len=max_source_len)
        
        entropy_ma = labeled_data['entropy_ma']

        for l in range(min(len(entropy_ma), len(human_entropy_mean), max_len)):
            entropy_violation = False
            if entropy_ma[l] > human_entropy_mean[l] + std_margin * human_entropy_std[l]:
                entropy_violation = True
                upper_bound_violations += 1
                upper_bound_violation_arr[l] += 1

            elif entropy_ma[l] < human_entropy_mean[l] - std_margin * human_entropy_std[l]:
                entropy_violation = True
                lower_bound_violations += 1
                lower_bound_violation_arr[l] += 1
            
            if entropy_violation:
                entropy_violations+= 1
                entropy_violation_arr[l] += 1
    
            count += 1
            count_arr[l] += 1
        
    for l in range(max_len):
        if count_arr[l] == 0:
            continue
        
        entropy_violation_arr[l] /= count_arr[l]
        lower_bound_violation_arr[l] /= count_arr[l]
        upper_bound_violation_arr[l] /= count_arr[l]

    return {
        'entropy_violation_ratio': entropy_violations/count, 
        'upper_bound_violation_ratio': upper_bound_violations/count, 
        'lower_bound_violation_ratio': lower_bound_violations/count,
        'entropy_violation_arr': entropy_violation_arr,
        'lower_bound_violation_arr': lower_bound_violation_arr,
        'upper_bound_violation_arr': upper_bound_violation_arr,
        'count_arr': count_arr,
    }
