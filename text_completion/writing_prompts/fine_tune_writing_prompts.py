
import math
import sys
import os
import evaluate

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))
from entropy_aware_search.hf_utils import DataArguments, ModelArguments, get_tokenizer, get_model
from utils import get_writing_prompt_dataset, preprocess_logits_for_metrics, get_tokenized_prompt_dataset, get_compute_metrics_func

learning_rate = float(os.environ.get("LR", 1e-4))
if __name__ == '__main__':
    per_device_batch_size = 8

    model_args = ModelArguments(
        model_name_or_path="gpt2-large",
    )

    output_dir_suffix = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    trainer_args = TrainingArguments(
        output_dir=os.path.expanduser(f"~/scratch/ews/finetuned_writing_prompts/{output_dir_suffix}/"),
        report_to="tensorboard",
        evaluation_strategy="steps",
        eval_steps=10,
        dataloader_num_workers=10,
        logging_steps=10,
        save_steps=1000,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=16,
        learning_rate=learning_rate,
        fp16=True,
        gradient_accumulation_steps=4,
    )

    prompt_response_dataset = get_writing_prompt_dataset("/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/")

    tokenizer = get_tokenizer(model_args)
    tokenizer.pad_token = tokenizer.eos_token

    model = get_model(model_args)

    prompt_dataset = get_tokenized_prompt_dataset(prompt_response_dataset, tokenizer)

    train_dataset = prompt_dataset["train"]
    eval_dataset = prompt_dataset["validation"]
    test_dataset = prompt_dataset["test"]

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    compute_metrics = get_compute_metrics_func(metric_names=['accuracy', 'mauve',], experiment_id=output_dir_suffix)
    
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

    metrics = trainer.evaluate()

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