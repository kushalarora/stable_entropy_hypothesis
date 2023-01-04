import sys
import os
import json

writing_prompt_dir = sys.argv[1]

for split in ('train', 'valid', 'test'):
    with open(os.path.join(writing_prompt_dir, f'{split}.wp_source')) as source_file, \
            open(os.path.join(writing_prompt_dir, f'{split}.wp_target')) as target_file, \
            open(os.path.join(writing_prompt_dir, f'{split}.json'), 'w') as output_file:

        sources = []
        targets = []
        data_dicts = []
        for source in source_file:
            sources.append(source)
        
        for target in target_file:
            targets.append(target)



        for (source, target) in zip(sources, targets):
            source = source.strip()
            target = target.strip()
            if not source.startswith("[ WP ]"):
                continue

            if len(source.split()) < 15:
                continue
            
            source = f"{source}\nStory: "
            data_dicts.append({
                'prompt': source, 
                'response': target.replace(" <newline> <newline>", "\n"),
            })
        
        for data_dict in data_dicts:
            print(json.dumps(data_dict), file=output_file)