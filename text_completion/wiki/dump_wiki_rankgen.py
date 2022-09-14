from torch.utils.data import DataLoader

from utils import get_wiki_dataset
import json

wiki_dataset = get_wiki_dataset("/home/mila/a/arorakus/wdir/entropy_aware_search/data/rankgen_data/")

with open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/wiki_rankgen/generated/orig.txt', 'w') as orig_file:
    for datapoint in wiki_dataset:
        prompt_sequence = datapoint['prefix'].strip().replace("\n", "<n>").replace("\t", "<tab>").replace('"', '\\"')
        generated_sequence = datapoint['targets'][0].strip().replace("\n", "<n>").replace("\t", "<tab>").replace('"', '\\"')
        print(f"{prompt_sequence}\t{generated_sequence}", file=orig_file)