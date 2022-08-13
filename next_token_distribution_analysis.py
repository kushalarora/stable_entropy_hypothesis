from typing import List, Union


import copy
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F



def load_model(modelname: str = None):
    if model_filepath is not None:
        model = create_agent_from_model_file(model_filepath)
    return model


def predict(model, context: str, model_text: str, width=1) -> Message:
    """
    Use model to predict given the context.

    :param context:
        The input context to classify.

    :return output:
        return output from model act.
    """
    assert isinstance(model, Agent)

    obs = Message({
        "model_label": "pos", # HACK: This is just a dummy label.
        "text":  context,
        "labels":  [model_text],
        "episode_done": False,
    })
    
    obs = model.observe(obs)
    model.self_observe(obs)

    batch = model.batchify([obs])
    batch = batch.to(next(model.model.parameters()).device)
    outputs = model.model(*model._model_input(batch), ys=batch.label_vec)
    
    model_scores = outputs[0]
    
    entropy = entropy_from_scores(model_scores)
    
    target_tokens = batch.label_vec
    next_tokens = target_tokens

    model_probs = F.softmax(model_scores, dim=-1)
    next_token_probs = model_probs\
                        .gather(-1, next_tokens.unsqueeze(-1))\
                        .squeeze(-1)
    
    model_probs_sorted_vals, model_probs_sorted_idxs = \
        torch.sort(model_probs, descending=True)
    
    top5_tokens = []
    for i in range(len(batch.label_vec[0])):
        top5_tokens.append([])
        for j, x in enumerate(model_probs_sorted_idxs[0, i, :5]):
            topk_token = model.dict.ind2tok[int(x)].replace("Ġ", " ")
            top5_tokens[i].append((topk_token,                             
                float(model_probs_sorted_vals[0, i, j])))
    
    label_toks = []
    for x in batch.label_vec[0].cpu():
        tok = model.dict.ind2tok[int(x)]
        label_toks.append(tok.replace("Ġ", " "))
        # label_toks.append(tok)

    context_toks = []
    for x in batch.text_vec[0].cpu():
        tok = model.dict.ind2tok[int(x)]
        context_toks.append(tok.replace("Ġ", " "))
        # context_toks.append(tok)

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

    ctxt_tok_str = "".join(context_toks)
    model_txt_tok_str = "".join(label_toks)
    (num_3_gram_repeat, creps_num_3_gram_repeat, 
        lreps_num_3_gram_repeat, rep_idxs, crep_idxs, lrep_idxs) = \
            compute_ngram_repeats(context_toks, label_toks, splitted=True)

    has_3_gram_repeat = num_3_gram_repeat > 0
    has_3_gram_label_repeat = lreps_num_3_gram_repeat > 0
    
    return {
        'tokens': label_toks,
        'context_tokens': context_toks,
        "probs": next_token_probs[0].cpu().tolist(),
        "top-5_tokens": top5_tokens,
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




if __name__ == '__main__':
    parser = ParlaiParser(False, False)
    parser.add_argument('-o', '--output-file', type=str, required=True, help='Output filepath.')
    parser.add_argument('-m', '--model-filepath', type=str, required=True, help='Model path.')
    parser.add_argument('-i', '--input-file', type=str, required=True, help='Input filepath')

    opt = parser.parse_args()

    model_filepath = opt['model_filepath']
    dataset_filepath = opt['input_file']
    output_filepath = opt['output_file']

    model = load_model(model_filepath)

    dataframe = parse_world_log_metadatafile(dataset_filepath)

    with open(output_filepath, 'w') as outfile:
        for idx, datapoint in dataframe.iterrows():
            labeled_datapoints = process_datapoint(
                model=model, datapoint=datapoint,
            )

            # labeled_tokens = [x[0] + '[' + x[1] + ']' for x in labeled_datapoints]
            # labeled_str = ' '.join(labeled_tokens)
            # output_str = (f"Eval Label:{datapoint.get('eval_label')}\t" 
            #               f"Context:{datapoint.get('context')}\t"
            #               f"Model text:{labeled_str}")
            # print(output_str, file=outfile)
            
            
