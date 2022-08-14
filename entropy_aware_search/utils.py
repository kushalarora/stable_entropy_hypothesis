from typing import List, Union


import copy
import json
import pandas as pd
import numpy as np

from termcolor import colored

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

def process_datapoint(model, datapoint, width=1):
    model.reset()
    "Returns list of tuples, with item at idx t, (token_t, label[token_t])."
    outputs = []
    model_text = datapoint['model_text'].strip()
    context = datapoint['context'].strip()
    model_act = predict(model=model, context=context, model_text=model_text, width=width)
    # target_act = predict(model=model, context=context, model_text=eval_label, is_edc=True)

    for idx, (token, is_rep, is_lrep, is_crep, entropy, entropy_ma, 
              dentropy, ddentropy, dddentropy, 
              prob, top5) in enumerate(zip(model_act['tokens'],
                                model_act['rep_idxs'],
                                model_act['lrep_idxs'],
                                model_act['crep_idxs'],
                                model_act['entropy'],
                                model_act['entropy_ma'],
                                model_act['dentropy'],
                                model_act['ddentropy'],
                                model_act['dddentropy'],
                                model_act['probs'], 
                                model_act['top-5_tokens'],)):
        if token == '__end__':
            break
        outputs.append((idx, token, is_rep, is_lrep, is_crep, entropy, 
                        entropy_ma, dentropy, ddentropy,  dddentropy, prob, top5))
    return outputs


def compute_average_across_sequences(dataframe, model, column_prefix, max_len=128, num_seq=500, to_be_averaged='entropy', width=1):
    counts = [0] * max_len
    cumls = [0.] * max_len
    values = np.zeros((num_seq, max_len))
    values.fill(-1)

    for j, (_,datapoint) in enumerate(dataframe.sample(num_seq).iterrows()):
        labeled_datapoint = process_datapoint(model=model, datapoint=datapoint, width=width)

        labeled_dataframe = pd.DataFrame(labeled_datapoint, columns=['index', 'token', 'is_rep', 'is_lrep', 'is_crep', 'entropy', 'entropy_ma', 'dent', 'ddent', 'dddent', 'probs', 'top-5'])
        
        for i,ent in labeled_dataframe[to_be_averaged].iteritems():
            if i >= max_len:
                break
            
            values[j, i] = ent
            cumls[i] += ent
            counts[i] += 1

    avgs = [cuml_ent/count if count > 0 else 0
                   for (cuml_ent,count) in zip(cumls, counts)]
    avgs_pd = pd.DataFrame(avgs,  columns=[column_prefix + '_' + to_be_averaged])
    
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
