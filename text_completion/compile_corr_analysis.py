import argparse
import os
import json
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--directory', default="/home/mila/a/arorakus/wdir/entropy_aware_search/data/wiki_rankgen/corr_analysis/gpt2_xl/")
args = parser.parse_args()
all_filtered_results = []
keys_to_filter = ['repeat_score@5', 'ngram_repeat@3', 'f1_score', 'dataset', 'entropy_violation_ratio',  'num_generations',
                           'avg_compute_time_in_secs', "upper_bound_violation_ratio", "lower_bound_violation_ratio", "mauve"]

for filename in os.listdir(args.directory):
    if not filename.endswith('.score'):
        continue

    print(filename)

    filepath = os.path.join(args.directory, filename)
    with open(filepath, 'r') as f:
        results = json.load(f)
        filtered_results = {}
        for key in keys_to_filter:
            filtered_results[key] = results[key]

        all_filtered_results.append(filtered_results)


df = pd.DataFrame(all_filtered_results)
print(df)

df.to_csv(os.path.join(args.directory, "compiled_results.csv"))
