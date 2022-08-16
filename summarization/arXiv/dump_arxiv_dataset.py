from torch.utils.data import DataLoader
from datasets import load_dataset

arxiv_summ_dataset = load_dataset("scientific_papers", "arxiv")
with open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/arxiv_pegasus/generated/orig.txt', 'w') as orig_file:
    for datapoint in arxiv_summ_dataset['test']:
        article = datapoint['article'].strip().replace("\n", "<newline>")
        abstract = datapoint['abstract'].strip().replace("\n", "<newline>")
        print(f"{article}\t{abstract}", file=orig_file)