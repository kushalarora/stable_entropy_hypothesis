from entropy_aware_search.utils import compute_average_across_sequences
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM

import argparse
import numpy as np
import pandas as pd
import json


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='/home/mila/a/arorakus/wdir/entropy_aware_search/data/wiki_rankgen/generated/orig.jsonl')
parser.add_argument('--max_len', default=128, type=int)
parser.add_argument('--width', default=5, type=int)
parser.add_argument('--max_num_seq', default=10000, type=int)
parser.add_argument('--model_name_or_path', default="gpt2-xl")
parser.add_argument('--degree', default=1, type=int)
parser.add_argument('--std_dev', default=1.0, type=float)
parser.add_argument('--is_seq2seq', action="store_true",)
parser.add_argument('--max_source_len', default=128, type=int)

args = parser.parse_args()

# get model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

if args.is_seq2seq:
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

model = model.to("cuda")

human_dataframe = pd.read_json(args.dataset, lines=True)

num_seq = min(args.max_num_seq, len(human_dataframe) - 1)

_, human_ma_entropies = compute_average_across_sequences(
                            human_dataframe, model, 
                            tokenizer,                            
                            column_prefix='human_generated', width=args.width,  max_len=args.max_len, 
                            to_be_averaged='entropy_ma', num_seq=num_seq, cache=True, is_seq2seq=args.is_seq2seq, max_source_len=args.max_source_len)

# _, human_entropies = compute_average_across_sequences(
#                             human_dataframe, model, 
#                             tokenizer,                            
#                             column_prefix='human_generated', width=args.width,  max_len=args.max_len, 
#                             to_be_averaged='entropy', num_seq=num_seq, cache=True, is_seq2seq=args.is_seq2seq, max_source_len=args.max_source_len)


# human_mean = np.ma.mean(human_entropies, axis=0)
# human_std = np.ma.std(human_entropies, axis=0)

human_ma_mean = np.ma.mean(human_ma_entropies, axis=0)
human_ma_std = np.ma.std(human_ma_entropies, axis=0)

seq_len = human_ma_mean.shape[0]

upper_cutoffs = human_ma_mean + args.std_dev * human_ma_std
lower_cutoffs = human_ma_mean - args.std_dev * human_ma_std

human_ma_mean_coeffs, human_ma_mean_fit_loss, *_ =  np.polyfit(np.arange(seq_len), 
                                                        human_ma_mean, deg=args.degree, full=True)
# human_mean_coeffs, human_mean_fit_loss, *_ =  np.polyfit(np.arange(seq_len), 
#                                                 human_mean, deg=args.degree, full=True)

human_ma_std_coeffs, human_ma_std_fit_loss, *_ =  np.polyfit(np.arange(seq_len), 
                                                    human_ma_std, deg=args.degree, full=True)
# human_std_coeffs, human_std_fit_loss, *_ =  np.polyfit(np.arange(seq_len), 
#                                                 human_std, deg=args.degree, full=True)


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
# print(f"\t human_mean_coeffs: {tuple([round(x, 5) for x in human_mean_coeffs])}")
print() 
# print(f"\t human_std_coeffs: {tuple([round(x, 5) for x in human_std_coeffs])}")
print()
print(f"\t lower_cutoff_coeffs: {tuple([round(x, 5) for x in upper_cutoff_coeffs])}")
print() 
print(f"\t upper_cutoff_coeffs: {tuple([round(x, 5) for x in lower_cutoff_coeffs])}")
print()
print()
print(f"Mean and Std. fit loss:")
print(f"human_ma_mean_fit_loss: {float(human_ma_mean_fit_loss/seq_len):.3f}")
print(f"human_ma_std_fit_loss: {float(human_ma_std_fit_loss/seq_len):.3f}")
# print(f"human_mean_fit_loss: {float(human_mean_fit_loss/seq_len):.3f}")
# print(f"human_std_fit_loss: {float(human_std_fit_loss/seq_len):.3f}")
print(f"upper_cutoff_fit_loss: {float(upper_cutoff_fit_loss/seq_len):.3f}")
print(f"lower_cutoff_fit_loss: {float(lower_cutoff_fit_loss/seq_len):.3f}")

coeffs = {
    "degree": args.degree,
    "human_ma_mean_coeffs": tuple([round(x, 5) for x in human_ma_mean_coeffs]),
    "human_ma_std_coeffs": tuple([round(x, 5) for x in human_ma_std_coeffs]),
    "lower_cutoff_coeffs": tuple([round(x, 5) for x in upper_cutoff_coeffs]),
    "upper_cutoff_coeffs": tuple([round(x, 5) for x in lower_cutoff_coeffs]),
    "human_ma_mean_fit_loss": round(float(human_ma_mean_fit_loss/seq_len), 5),
    "human_ma_std_fit_loss": round(float(human_ma_std_fit_loss/seq_len),5),
    "upper_cutoff_fit_loss": round(float(upper_cutoff_fit_loss/seq_len), 5),
    "lower_cutoff_fit_loss": round(float(lower_cutoff_fit_loss/seq_len), 5),
}

output_filename=f"{args.dataset}.deg_{args.degree}.coeffs.json"
with open(output_filename, 'w') as coeff_file:
    json.dump(coeffs, coeff_file, indent=4)
print(f"Coeffs written to: {output_filename}")




