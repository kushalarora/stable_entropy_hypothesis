
import evaluate

import os
from datasets import load_dataset

def get_wiki_dataset(data_dir):
    
    wiki_dataset = load_dataset("json", 
                    data_files=os.path.join(data_dir, 'wiki.jsonl'))
    return wiki_dataset['train']

def get_pg19_dataset(data_dir):
    
    wiki_dataset = load_dataset("json", 
                    data_files=os.path.join(data_dir, 'pg19.jsonl'))
    return wiki_dataset['train']

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

def get_compute_metrics_func(experiment_id, metric_names=['accuracy'], tokenizer=None):

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    metrics = []
    if "accuracy" in metric_names:
        accuracy = evaluate.load("accuracy", 
                                experiment_id=experiment_id, 
                                cache_dir=f"/tmp/{experiment_id}")
        metrics.append((accuracy, False))
    
    if "mauve" in metric_names:
        mauve = evaluate.load("mauve",
                               experiment_id=experiment_id, 
                               cache_dir=f"/tmp/{experiment_id}")
        metrics.append(mauve)

    def compute_metrics(eval_preds):
        output = {}
        import pdb; pdb.set_trace()
        for metric, needs_decoding in metrics:
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)

            if needs_decoding:
                preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
                labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                preds, labels = postprocess_text(preds, labels)
            output.update(metric.compute(predictions=preds, references=labels))
        return output
    return compute_metrics