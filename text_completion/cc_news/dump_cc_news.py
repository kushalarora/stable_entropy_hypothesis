from datasets import load_dataset
import json

cc_news_dataset = load_dataset("cc_news", split='train')

with open('data/cc_news/generated/orig.jsonl', 'w') as orig_file:
    for datapoint in cc_news_dataset:
        prompt_sequence = datapoint['title']
        generated_sequence = ' '.join(datapoint['text'].split()[:512])
        output = {
            'context': prompt_sequence,
            'model_text': generated_sequence,
        }
        print(json.dumps(output), file=orig_file)