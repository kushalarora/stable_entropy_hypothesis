from utils import get_writing_prompt_dataset
import json


writing_prompt_dataset = get_writing_prompt_dataset("/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/")

with open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/generated/orig.jsonl', 'w') as orig_file:
    for datapoint in writing_prompt_dataset['test']:
        prompt_sequence = datapoint['prompt']

        generated_sequence = datapoint['response']
        output = {
            'context': prompt_sequence,
            'model_text': generated_sequence,
        }
        print(json.dumps(output), file=orig_file)
