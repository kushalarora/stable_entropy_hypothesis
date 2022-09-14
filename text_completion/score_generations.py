import argparse
import json
import random
import numpy as np
import mauve
import pickle
import glob
import os
import matplotlib.pyplot as plt
from nltk import tokenize
from nltk.corpus import stopwords
import tqdm
import re 
import string
import collections as cll
import spacy

nlp = spacy.load("en_core_web_md")

def f1_score(prediction, ground_truth, gram=1, stopwords=None):
    """Calculate word level F1 score."""
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    prediction_tokens = [
        " ".join(prediction_tokens[i:i + gram])
        for i in range(0, len(prediction_tokens) - gram + 1)
    ]
    ground_truth_tokens = [
        " ".join(ground_truth_tokens[i:i + gram])
        for i in range(0, len(ground_truth_tokens) - gram + 1)
    ]

    if stopwords:
        prediction_tokens = [x for x in prediction_tokens if x not in stopwords]
        ground_truth_tokens = [x for x in ground_truth_tokens if x not in stopwords]

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0, 1.0, 1.0
    common = cll.Counter(prediction_tokens) & cll.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def rep_statistic(prefix, suffix, window=20):
    prefix_tokens = normalize_answer(prefix).split()
    suffix_tokens = normalize_answer(suffix).split()
    start_pos = len(prefix_tokens)
    tokens = prefix_tokens + suffix_tokens
    reps = [tokens[i] in tokens[i - window:i] for i in range(start_pos, len(tokens))]
    if len(reps) == 0:
        return 0.0
    else:
        return np.mean(reps)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="/home/mila/a/arorakus/wdir/entropy_aware_search/data/wiki_rankgen/generated/gpt2_xl/greedy.csv")
parser.add_argument('--eval_type', default="max")
parser.add_argument('--gram', default=1, type=int)
parser.add_argument('--rep_window', default=20, type=int)
parser.add_argument('--plot_divergence', action='store_true')
parser.add_argument('--eval_mauve', action='store_true')
args = parser.parse_args()




with open(args.dataset, 'r') as generations:
    generated_seqs = []
    human_seqs = []
    for line in generations:
        data = json.loads(line.strip())

        prefix = data['prefix'].strip()
        generated_seq = data['generation'].strip()
        target = data['target'].strip()

        # import pdb; pdb.set_trace()

        # generated_seq = ' '.join([str(x) for x in nlp(generated_seq).sents])
        # target = ' '.join(nlp(target).sents)

        generated_seqs.append(generated_seq)
        human_seqs.append(target)
        # generated_seqs.append(target)
        # human_seqs.append(prefix)
        # generated_seqs.append(' '.join((prefix, generated_seq)))
        # human_seqs.append(' '.join((prefix, target)))

    mauve = mauve.compute_mauve(p_text=generated_seqs, q_text=human_seqs, device_id=0, verbose=True, batch_size=16, max_text_length=768,)
    print(f"Mauve Score = {mauve}")
    outputs = {
        "mauve": mauve.mauve,
    }
    with open(args.dataset + ".score", "w") as score_file:
        print(json.dumps(outputs, indent=4), file=score_file)
