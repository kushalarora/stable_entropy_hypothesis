import argparse
import json
import random
import numpy as np
import mauve
import pickle
import glob
import os
# import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import tqdm
import re 
import string
import collections as cll
# import spacy
import math
import pickle
# nlp = spacy.load("en_core_web_md")

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

def compute_ngram_repeats(prefix, generated, ngram_sz):
    cgrams = {}
    # compute N grams of the context
    text = prefix.split()
    for i in range((ngram_sz-1), len(text)):
        ngram = ' '.join(text[i - (ngram_sz-1) : i + 1])
        cgrams[ngram] = True

    # compute N grams of the model response
    creps = 0
    lreps = 0

    text = generated.split()
    lgrams = {}
    for i in range((ngram_sz-1), len(text)):
        ngram = ' '.join(text[i - (ngram_sz-1) : i + 1])
        if ngram in cgrams:
            creps = creps + 1
        else:
            if ngram in lgrams:
                lreps = lreps + 1
        lgrams[ngram] = True
    return creps + lreps


def repeat_score(prefix, generated, upto_ngrams=5):
    cuml_rep_score = 0
    cuml_reps = 0
    _1gram_reps = 0
    ngram_repeats = []
    for n in range(1, upto_ngrams+1):
        reps = compute_ngram_repeats(prefix, generated, n)
        if n == 1:
            _1gram_reps = reps

        ngram_repeats.append(reps)
        cuml_rep_score += math.pow(2, n) * reps
        cuml_reps += reps
    avg_repeat_len = 0

    if cuml_rep_score > 0:
        avg_repeat_len = math.log(cuml_rep_score/cuml_reps, 2)

    repeat_score = (avg_repeat_len * _1gram_reps)/len(generated.split())
    return repeat_score, avg_repeat_len, ngram_repeats


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
    token_overlaps = []
    repeat_scores = []
    avg_rep_lens = []
    ngram_repeats = {1: [], 2: [], 3: [], 4:[], 5:[]}
    for line in generations:
        data = json.loads(line.strip())

        prefix = data['prefix']
        generated_seq = data['generation'].strip()
        target = data['target'].strip()
        
        if generated_seq is None or len(generated_seq.split()) < 10 or \
            target is None or len(target.split()) < 10:
            continue

        # import pdb; pdb.set_trace()

        # generated_seq = ' '.join([str(x) for x in nlp(generated_seq).sents])
        # target = ' '.join(nlp(target).sents)

        generated_seqs.append(prefix + " " + generated_seq)
        human_seqs.append(prefix + " " + target)

        token_overlaps.append(f1_score(generated_seq, target, stopwords=stopwords.words('english'), gram=args.gram)[-1])
        rep_score, avg_rep_len, ngram_repeat = repeat_score(prefix, generated_seq, 5)
        repeat_scores.append(rep_score)
        avg_rep_lens.append(avg_rep_len)

        for i, rep_n in enumerate(ngram_repeat):
            ngram_repeats[i+1].append(rep_n)

    outputs = {
        "f1_score":  np.mean(token_overlaps),
        "repeat_score@5": np.mean(repeat_scores),
        "avg_rep_lens@5": np.mean(avg_rep_lens),
    }
    for i, ngs in ngram_repeats.items():
        outputs[f'ngram_repeat@{i}'] = np.mean(ngs)

    mauve_filename = args.dataset + ".mauve"
    generated_seq_hash = abs(hash(args.dataset)) + abs(hash(" ".join(generated_seqs)))
    
    compute_mauve_score = not os.path.exists(mauve_filename)
    if not compute_mauve_score:
        with open(mauve_filename, 'rb') as mauve_file:
            mauve_score_dict = pickle.load(mauve_file)
            if mauve_score_dict['hash'] != generated_seq_hash:
                compute_mauve_score = True

    if compute_mauve_score:
        mauve_score = mauve.compute_mauve(p_text=generated_seqs, q_text=human_seqs, device_id=0, verbose=True, batch_size=16, max_text_length=768,)
        mauve_score_dict = vars(mauve_score)
        mauve_score_dict['hash'] = generated_seq_hash

        with open(mauve_filename, 'wb') as mauve_file:
            mauve_score = pickle.dump(mauve_score_dict, mauve_file)
    
    print(f"Dataset: {args.dataset}")
    print(f"Mauve Score = {mauve_score_dict['mauve']}")
    outputs["mauve"] = mauve_score_dict['mauve']

    with open(args.dataset + ".score", "w") as score_file:
        print(json.dumps(outputs, indent=4), file=score_file)
