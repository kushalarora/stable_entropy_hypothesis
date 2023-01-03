from datasets import load_dataset
import json


bst = load_dataset("blended_skill_talk")
with open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/blended_skill_talk/generated/orig.jsonl', 'w') as orig_file:
    for datapoint in bst['test']:
        previous_utterance = datapoint['previous_utterance']
        free_messages = datapoint['free_messages']
        guided_messages = datapoint['guided_messages']
        
        previous_utterances = previous_utterance
        new_utterances = []
        for i, utt in enumerate(previous_utterances):
            # utt = f"Person {i % 2 + 1}: {utt}"
            new_utterances.append(utt + "\n")
        previous_utterances = new_utterances

        for free,guided in zip(free_messages, guided_messages):
            for i, utt in enumerate((free, guided)):
                context = ''.join(previous_utterances)
                j = 0
                while len(context.split()) > 80:
                    j += 1
                    context = ''.join(previous_utterances[j:])

                if len(context.split()) < 2 or len(utt.split()) < 2:
                    continue
                
                # context += f"Person {i % 2 + 1}:"
                output = {
                    'context': context,
                    'model_text': utt,
                }
                print(json.dumps(output), file=orig_file)
                # utt = f"Person {i % 2 + 1}: {utt}"
                previous_utterances.append(utt + "\n")
