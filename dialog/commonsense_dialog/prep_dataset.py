import sys
import os
import json

commonsense_dialog = sys.argv[1]

for split in ('train', 'valid', 'test'):
    with open(os.path.join(commonsense_dialog, f'{split}.json')) as source_file, \
            open(os.path.join(commonsense_dialog, f'{split}_prepped.json'), 'w') as output_file:

        raw_data_dict = json.load(source_file)

        for key,value in raw_data_dict.items():
            turns = value['turns']
            previous_utterances = turns[0:1]
            for turn in turns[1:]:
                context = '\n'.join(previous_utterances)
                output = {
                    'context': context,
                    'model_text': turn,
                }
                previous_utterances.append(turn)

                print(json.dumps(output), file=output_file)