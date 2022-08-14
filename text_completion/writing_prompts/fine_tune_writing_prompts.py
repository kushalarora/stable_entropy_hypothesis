
import math
import sys
import os
import evaluate

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    DataCollatorForLanguageModeling
)

from datasets import load_dataset
from datetime import datetime

from transformers import default_data_collator
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))
from entropy_aware_search.hf_utils import DataArguments, ModelArguments, get_tokenizer, get_model

from torch.utils.data import DataLoader

# Preprocessing inspired from 
# https://github.com/anshradh/trl_custom/blob/master/nbs/03_writing_prompt_reward_model_training.ipynb

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

def get_tokenized_prompt_dataset(prompt_response_dataset, tokenizer, generate=False):
    def preprocess_and_tokenize(examples):
        prompts = []
        responses = []
        if generate:
            tokenized_examples = tokenizer(examples['prompt'])
        else:
            tokenized_examples = tokenizer(examples['prompt'], examples['response'], max_length=1024, truncation=True)
            # tokenized_examples['labels'] = tokenized_examples['input_ids'].copy()
        return tokenized_examples

    prompt_dataset = prompt_response_dataset.map(
                        preprocess_and_tokenize, 
                        batched=True, 
                        num_proc=10, 
                        remove_columns=prompt_response_dataset['validation'].column_names)
    return prompt_dataset

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


if __name__ == '__main__':
    per_device_batch_size = 8

    model_args = ModelArguments(
        model_name_or_path="gpt2",
    )

    output_dir_suffix = datetime.now().strftime("%m-%d-%Y-%H-%M")
    trainer_args = TrainingArguments(
        output_dir=os.path.expanduser(f"~/scratch/ews/finetuned_writing_prompts/{output_dir_suffix}/"),
        report_to="tensorboard",
        evaluation_strategy="steps",
        eval_steps=500,
        dataloader_num_workers=10,
        logging_steps=10,
        save_steps=1000,
    )

    prompt_response_dataset = get_writing_prompt_dataset("/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/")
    tokenizer = get_tokenizer(model_args)
    tokenizer.pad_token = tokenizer.eos_token

    prompt_dataset = get_tokenized_prompt_dataset(prompt_response_dataset, tokenizer)

    model = get_model(model_args)

    train_dataset = prompt_dataset["train"]
    eval_dataset = prompt_dataset["validation"]
    test_dataset = prompt_dataset["test"]

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    metric = evaluate.load("accuracy")

    # Initialize our Trainer
    trainer = Trainer(
        args=trainer_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate()

    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)