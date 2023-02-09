
import argparse
import numpy as np
import pandas as pd
import json


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='/home/mila/a/arorakus/wdir/entropy_aware_search/data/wiki_rankgen/generated/orig.jsonl')
args = parser.parse_args()

pct_violations = []
num_backoffs = []

with open(args.dataset) as file:
    for line in file:
        obj =  json.loads(line)
        pct_violation = obj["pct_upper_violations"]
        if np.isnan(pct_violation):
            continue

        pct_violations.append(pct_violation)
        num_backoffs.append(obj["num_backoffs"])


print(f"Percent Greedy: {(100 - np.mean(pct_violations))}")
print(f"Num Backoffs: {np.mean(num_backoffs)/4}")