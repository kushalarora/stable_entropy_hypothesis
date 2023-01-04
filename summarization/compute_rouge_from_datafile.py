from datetime import datetime
import sys
import json
import evaluate

experiment_id = "pegasus_bigbird_arxiv_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

rouge = evaluate.load("rouge", experiment_id=experiment_id, 
                        cache_dir=f"/tmp/{experiment_id}.txt")

with open(sys.argv[1]) as datafile:
    rouges_dict = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}

    generated_abstracts = []
    targets = []
    for line in datafile:
        dict = json.loads(line)
        generated_abstract = dict["generation"]
        target = dict['target']

        generated_abstracts.append(generated_abstract)
        targets.append(target)


    rouge_scores = rouge.compute(predictions=generated_abstracts, 
         references=targets, use_stemmer=True)

    print(rouge_scores)