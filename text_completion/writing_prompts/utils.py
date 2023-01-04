
import evaluate

import os
from datasets import load_dataset

def get_writing_prompt_dataset(data_dir):
    data_files = {
        'train': os.path.join(data_dir, 'train.json'),
        'validation': os.path.join(data_dir, 'valid.json'),
        'test': os.path.join(data_dir, 'test.json')
    }
    prompt_response_dataset = load_dataset("json", 
                                            data_files=data_files)
    return prompt_response_dataset



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