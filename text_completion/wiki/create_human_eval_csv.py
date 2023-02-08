import sys
import argparse 
import json
import hashlib
import random
import pandas as pd

parser = argparse.ArgumentParser()


parser.add_argument("--filenames", type=str, nargs='+')
parser.add_argument("--decoding_methods", type=str, nargs='+')
parser.add_argument("--output_filename", type=str)

args = parser.parse_args()

assert len(args.filenames) == 2

method2data = {}
for method, filename in zip(args.decoding_methods, args.filenames):
    with open(filename) as file:
        for idx, line in enumerate(file):
            jsonobj = json.loads(line)
            context = jsonobj['prefix']
            context = ' '.join(context.split()[-50:])
            context = '...' + context.replace("\n", "<n>")
            generation = jsonobj['generation'].replace("\n", "<n>")
            id = idx # str(hash(context))[:12]
            if id not in method2data:
                method2data[id] = {
                    'context': context,
                    method: generation
                }
            else:
                assert method2data[id]['context'] == context
                method2data[id][method] = generation


idxs = random.choices(range(len(method2data)), k=100)

selected_datapoints = []
selected_datapoints_anonymized = []
decoding_methods = list(args.decoding_methods)
for idx in idxs:
    context = method2data[idx]['context']
    random.shuffle(decoding_methods)
    methods_and_generations = []
    for method in decoding_methods:
        methods_and_generations.append((method, method2data[idx][method]))
    method1 = methods_and_generations[0][0]
    generation1 = methods_and_generations[0][1]
    method2 = methods_and_generations[1][0]
    generation2 = methods_and_generations[1][1]
    selected_datapoints.append((context, method1, method2, generation1, generation2))
    selected_datapoints_anonymized.append((context, generation1, generation2))

df_non_anonymized = pd.DataFrame(selected_datapoints, columns=['Context', 'Method 1', 'Method 2', 'Generation 1', 'Generation 2'])
df_anonymized = pd.DataFrame(selected_datapoints_anonymized, columns=['Context', 'Generation 1', 'Generation 2'])

df_non_anonymized.to_csv(f'{args.output_filename}.non_anonymized.csv')
df_anonymized.to_csv(f'{args.output_filename}.anonymized.csv')
