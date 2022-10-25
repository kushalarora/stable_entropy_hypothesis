from entropy_aware_search.hf_utils import ModelArguments, get_tokenizer, get_model
from entropy_aware_search.utils import compute_average_across_sequences

import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='/home/mila/a/arorakus/wdir/entropy_aware_search/data/wiki_rankgen/generated/orig.jsonl')
parser.add_argument('--max_len', default=128, type=int)
parser.add_argument('--width', default=5, type=int)
parser.add_argument('--max_num_seq', default=10000, type=int)
parser.add_argument('--model_name_or_path', default="gpt2-xl")
parser.add_argument('--degree', default=1, type=int)
parser.add_argument('--std_dev', default=1.0, type=float)

args = parser.parse_args()

# get model and tokenizer.
model_args = ModelArguments(
    model_name_or_path=args.model_name_or_path)
gpt2_model = get_model(model_args)
gpt2_model.to('cuda')
tokenizer = get_tokenizer(model_args)
tokenizer.pad_token = tokenizer.eos_token
gpt2_model = gpt2_model.to('cuda')

human_dataframe = pd.read_json(args.dataset, lines=True)\
                    .rename(columns={'prefix':'context', 
                                    'target': 'model_text'})


num_seq = min(args.max_num_seq, len(human_dataframe))

_, human_ma_entropies = compute_average_across_sequences(
                            human_dataframe, gpt2_model, 
                            tokenizer,                            
                            column_prefix='human_generated', width=args.width,  max_len=args.max_len, 
                            to_be_averaged='entropy_ma', num_seq=num_seq, cache=True)

_, human_entropies = compute_average_across_sequences(
                            human_dataframe, gpt2_model, 
                            tokenizer,                            
                            column_prefix='human_generated', width=args.width,  max_len=args.max_len, 
                            to_be_averaged='entropy', num_seq=num_seq, cache=True)


human_mean = np.ma.mean(human_entropies, axis=0)
human_std = np.ma.std(human_entropies, axis=0)

human_ma_mean = np.ma.mean(human_ma_entropies, axis=0)
human_ma_std = np.ma.std(human_ma_entropies, axis=0)

seq_len = human_ma_mean.shape[0]

upper_cutoffs = human_mean + args.std_dev * human_std
lower_cutoffs = human_ma_mean - args.std_dev * human_ma_std

human_ma_mean_coeffs, human_ma_mean_fit_loss, *_ =  np.polyfit(np.arange(seq_len), 
                                                        human_ma_mean, deg=args.degree, full=True)
human_mean_coeffs, human_mean_fit_loss, *_ =  np.polyfit(np.arange(seq_len), 
                                                human_mean, deg=args.degree, full=True)

human_ma_std_coeffs, human_ma_std_fit_loss, *_ =  np.polyfit(np.arange(seq_len), 
                                                    human_ma_std, deg=args.degree, full=True)
human_std_coeffs, human_std_fit_loss, *_ =  np.polyfit(np.arange(seq_len), 
                                                human_std, deg=args.degree, full=True)


upper_cutoff_coeffs, upper_cutoff_fit_loss, *_ = np.polyfit(np.arange(seq_len), 
                                                    upper_cutoffs, deg=args.degree, full=True)
lower_cutoff_coeffs, lower_cutoff_fit_loss, *_ = np.polyfit(np.arange(seq_len), 
                                                    lower_cutoffs, deg=args.degree, full=True)

print("Coeffs for Entropy Aware Search:")
print(f"Degree: {args.degree}")
print(f"\t human_ma_mean_coeffs: {tuple([round(x, 5) for x in human_ma_mean_coeffs])}")
print() 
print(f"\t human_ma_std_coeffs: {tuple([round(x, 5) for x in human_ma_std_coeffs])}")
print()
print(f"\t human_mean_coeffs: {tuple([round(x, 5) for x in human_mean_coeffs])}")
print() 
print(f"\t human_std_coeffs: {tuple([round(x, 5) for x in human_std_coeffs])}")
print()
print(f"\t lower_cutoff_coeffs: {tuple([round(x, 5) for x in upper_cutoff_coeffs])}")
print() 
print(f"\t upper_cutoff_coeffs: {tuple([round(x, 5) for x in lower_cutoff_coeffs])}")
print()
print()
print(f"Mean and Std. fit loss:")
print(f"human_ma_mean_fit_loss: {float(human_ma_mean_fit_loss):.3f}")
print(f"human_ma_std_fit_loss: {float(human_ma_std_fit_loss):.3f}")
print(f"human_mean_fit_loss: {float(human_mean_fit_loss):.3f}")
print(f"human_std_fit_loss: {float(human_std_fit_loss):.3f}")
print(f"upper_cutoff_fit_loss: {float(upper_cutoff_fit_loss):.3f}")
print(f"lower_cutoff_fit_loss: {float(lower_cutoff_fit_loss):.3f}")






