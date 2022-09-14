from torch.utils.data import DataLoader
from datasets import load_dataset

arxiv_summ_dataset = load_dataset("xsum")
with open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/xsum_pegasus/generated/orig.txt', 'w') as orig_file:
    for datapoint in arxiv_summ_dataset['test']:
        document = datapoint['document'].strip().replace("\n", "\\n")
        summary = datapoint['summary'].strip().replace("\n", "\\n")
        print(f"{document}\t{summary}", file=orig_file)