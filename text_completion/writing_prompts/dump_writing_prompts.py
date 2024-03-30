from utils import get_writing_prompt_dataset
import json


writing_prompt_dataset = get_writing_prompt_dataset()

import pdb; pdb.set_trace()
with open('data/writingPrompts/generated/orig.jsonl', 'w') as orig_file:
    for datapoint in writing_prompt_dataset['train']:
        prompt_sequence = datapoint['prompt']

        generated_sequence = datapoint['body']
        output = {
            'context': prompt_sequence,
            'model_text': generated_sequence,
        }
        print(json.dumps(output), file=orig_file)
