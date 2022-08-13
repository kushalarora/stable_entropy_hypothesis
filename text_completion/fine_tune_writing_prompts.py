
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
    DataCollatorWithPadding
)

from datasets import load_dataset
from datetime import datetime

from transformers import default_data_collator
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))
from entropy_beam_search.hf_utils import DataArguments, ModelArguments, get_tokenizer, get_model

# Preprocessing inspired from 
# https://github.com/anshradh/trl_custom/blob/master/nbs/03_writing_prompt_reward_model_training.ipynb
from torch.utils.data import DataLoader

per_device_batch_size = 8

model_args = ModelArguments(
    model_name_or_path="gpt2",
)


output_dir_suffix = datetime.now().strftime("%m-%d-%Y-%H-%M")
trainer_args = TrainingArguments(
    output_dir=os.path.expanduser(f"~/scratch/ews/finetuned_writing_prompts/{output_dir_suffix}/"),
    report_to="tensorboard",
    evaluation_strategy="steps",
    eval_steps=2000,
    dataloader_num_workers=10,
    fp16=True, 
    auto_find_batch_size=True,
    logging_steps=100,
)
prompt_response_dataset = load_dataset("rewardsignal/reddit_writing_prompts", 
                                        data_files="prompt_responses_full.csv")

prompt_response_dataset['train'] = load_dataset("rewardsignal/reddit_writing_prompts", 
                                        data_files="prompt_responses_full.csv", 
                                        split='train[:90%]')

prompt_response_dataset['validation'] = load_dataset(
                                            "rewardsignal/reddit_writing_prompts", 
                                            data_files="prompt_responses_full.csv", 
                                            split='train[90%:95%]')
prompt_response_dataset['test'] = load_dataset(
                                            "rewardsignal/reddit_writing_prompts", 
                                            data_files="prompt_responses_full.csv", 
                                            split='train[95%:100%]')

tokenizer = get_tokenizer(model_args)
tokenizer.pad_token = tokenizer.eos_token


prompt_prefix = "Writing Prompt: "
response_prefix = "Response: "

def preprocess_text_function(examples):
    prompts = []
    responses = []
    for prompt,response in zip(examples['prompt'], examples['response']):
        if prompt is None or response is None:
            prompt = ''
            response = ''
        prompt = prompt.replace('[WP] ', prompt_prefix)
        response = response_prefix + response 
        prompts.append(prompt)
        responses.append(response)
    examples["prompt"] = prompts
    examples["response"] = responses


    tokenized_examples = tokenizer(examples['prompt'], examples['response'], truncation=True, add_special_tokens=True, padding=True)
    tokenized_examples['labels'] = tokenized_examples['input_ids'].copy()
    return tokenized_examples

prompt_dataset = prompt_response_dataset.map(
                        preprocess_text_function, 
                        batched=True, 
                        num_proc=10, 
                        remove_columns=prompt_response_dataset['validation'].column_names)

model = get_model(model_args)

train_dataset = prompt_dataset["train"]
eval_dataset = prompt_dataset["validation"]
test_dataset = prompt_dataset["test"]

data_collator = DataCollatorWithPadding(tokenizer,)


# DataLoaders creation:
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_batch_size)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=per_device_batch_size)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)


# Initialize our Trainer
trainer = Trainer(
    args=trainer_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
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