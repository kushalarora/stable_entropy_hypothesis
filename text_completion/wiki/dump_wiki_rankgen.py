
from utils import get_wiki_dataset
import json

wiki_dataset = get_wiki_dataset("/home/mila/a/arorakus/wdir/entropy_aware_search/data/rankgen_data/")

with open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/wiki_rankgen/generated/orig.jsonl', 'w') as orig_file:
    for datapoint in wiki_dataset:
        prompt_sequence = datapoint['prefix']
        generated_sequence = datapoint['targets'][0]

        output = {
            'context': prompt_sequence,
            'model_text': generated_sequence,
        }
        print(json.dumps(output), file=orig_file)