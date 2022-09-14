from torch.utils.data import DataLoader
from datasets import load_dataset

from transformers import AutoTokenizer

summ_dataset = load_dataset("cnn_dailymail", "3.0.0")
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")

with open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/cnn_dm_pegasus/generated/orig.txt', 'w') as orig_file:
    for datapoint in summ_dataset['test']:
        document = datapoint['article'].strip().replace("\n", " <n> ")
        summary = datapoint['highlights'].strip().replace("\n", " <n> ")

        tokenized_document = tokenizer(document, max_length=1024, truncation=True, )
        tokenized_summary = tokenizer(summary, max_length=256, truncation=True, )

        document = tokenizer.decode(tokenized_document['input_ids'],  skip_special_tokens=True)
        summary = tokenizer.decode(tokenized_summary['input_ids'], skip_special_tokens=True)
        print(f"{document}\t{summary}", file=orig_file)
