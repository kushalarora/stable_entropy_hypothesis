from datasets import load_dataset
import json

summ_dataset = load_dataset("cnn_dailymail", "3.0.0")

with open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/cnn_dm_pegasus/generated/orig.jsonl', 'w') as orig_file:
    for datapoint in summ_dataset['test']:
        document = datapoint['article']
        summary = datapoint['highlights']

        output = {
            "context": document,
            "model_text": summary,
        }
        print(json.dumps(output), file=orig_file)
