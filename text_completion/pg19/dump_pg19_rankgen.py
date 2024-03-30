
from utils import get_pg19_dataset
import json

wiki_dataset = get_pg19_dataset("data/rankgen_data/")

with open('data/pg19_rankgen/generated/orig.jsonl', 'w') as orig_file:
    for datapoint in wiki_dataset:
        prompt_sequence = datapoint['prefix']
        generated_sequence = datapoint['targets'][0]

        if len(generated_sequence.split()) < 10:
            continue
        prompt_sequence = ' '.join(prompt_sequence.split()[256:])
        generated_sequence = ' '.join(generated_sequence.split()[:512])
        output = {
            'context': prompt_sequence,
            'model_text': generated_sequence,
        }
        print(json.dumps(output), file=orig_file)