from typing import List, Union


import copy
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from entropy_aware_search.utils import process_datapoint
from entropy_aware_search.hf_utils import ModelArguments,get_model,get_tokenizer

if __name__ == '__main__':


    gpt2_writing_prompt_modelfile = "/home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-14-2022-11-03/checkpoint-71000/"
    # gpt2_writing_prompt_modelfile = 'gpt2-large'
    model_args = ModelArguments(
        model_name_or_path=gpt2_writing_prompt_modelfile,   
    )
    gpt2_model = get_model(model_args)
    tokenizer = get_tokenizer(model_args)
    tokenizer.pad_token = tokenizer.eos_token

    orig_wp = '/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/generated/orig.txt'
    dataframe = pd.read_csv(orig_wp, sep='\t', names=['context', 'model_text'])

    datapoint = dataframe.sample(1).iloc[0]

    labeled_datapoints = process_datapoint(
            model=gpt2_model, tokenizer=tokenizer, datapoint=datapoint, max_len=1024
        )
