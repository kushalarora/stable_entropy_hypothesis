import argparse
import json
import numpy as np
from entropy_aware_search.utils import  compute_ngram_repeats
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="/home/mila/a/arorakus/wdir/entropy_aware_search/data/wiki_rankgen/generated/gpt2_xl/greedy.jsonl")
parser.add_argument('--gram', default=3, type=int)
parser.add_argument('--max_len', default=100, type=int)

args = parser.parse_args()


def index_till_degeneration(dataset, ngram, max_len):
    ngrams_till_t = []
    counts_till_t = []
    with open(dataset, 'r') as dataset_file:
        for line in dataset_file:
            data = json.loads(line)

            context = data['prefix']
            generation = data['generation']
            
            generated_tokens = generation.split()
            ngrams_till_t.append([0.] * max_len)
            counts_till_t.append([0.] * max_len)

            for i in range(min(len(generated_tokens), max_len)):
                prefix = ' '.join(generated_tokens[:i])
                _, _, reps, *_ = compute_ngram_repeats(context, prefix, ngram)
                ngrams_till_t[-1][i] = reps
                counts_till_t[-1][i] = 1

    ngrams_till_t_ma = np.ma.array(data=ngrams_till_t, 
                        mask=np.array(counts_till_t) == 0)

    return ngrams_till_t_ma

if __name__ == '__main__':
    ngrams_till_t = index_till_degeneration(args.dataset, 
                            args.gram, args.max_len)
    has_ngram_repeat_till_t = (ngrams_till_t > 0)
    mean_has_ngram_till_t = has_ngram_repeat_till_t.mean(0).data
    outputs = (
        ("till_0_idx", (mean_has_ngram_till_t == 0).sum()),
        ("till_0.01_idx", (mean_has_ngram_till_t < 0.01).sum()),
        ("till_0.05_idx", (mean_has_ngram_till_t < 0.05).sum()),
        ("till_0.1_idx", (mean_has_ngram_till_t < 0.1).sum()),
        ("till_0.25_idx", (mean_has_ngram_till_t < 0.25).sum()),
        ("till_0.50_idx", (mean_has_ngram_till_t < 0.5).sum()),
    )

    print(pd.DataFrame(outputs))



