
import math
import sys
import os
import evaluate

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM
)

from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))
from utils import get_writing_prompt_dataset, preprocess_logits_for_metrics, get_compute_metrics_func

learning_rate = float(os.environ.get("LR", 1e-4))
if __name__ == '__main__':
    per_device_batch_size = 4

    output_dir_suffix = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    trainer_args = TrainingArguments(
        output_dir=os.path.expanduser(f"~/scratch/ews/finetuned_writing_prompts/{output_dir_suffix}/"),
        report_to="tensorboard",
        evaluation_strategy="steps",
        warmup_steps=500,
        eval_steps=1000,
        dataloader_num_workers=10,
        logging_steps=100,
        save_steps=5000,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        fp16=True,
        # auto_find_batch_size=True,
        gradient_accumulation_steps=4,
    )

    prompt_response_dataset = get_writing_prompt_dataset("/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/")

    tokenizer_kwargs = {
        "cache_dir": "/home/mila/a/arorakus/scratch/huggingface/cache/",
        "use_fast": True,
    }
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large", **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2-large")
    model = model.to(0)

    def tokenize(examples):
        tokenized_examples = tokenizer(examples['prompt'], examples['response'], 
                                        max_length=1024, padding=True, truncation=True)
        return tokenized_examples

    prompt_dataset = prompt_response_dataset.map(
                        tokenize, 
                        batched=True,
                         num_proc=10, 
                        remove_columns=prompt_response_dataset['validation'].column_names)

    train_dataset = prompt_dataset["train"]
    eval_dataset = prompt_dataset["validation"]
    test_dataset = prompt_dataset["test"]

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    compute_metrics = get_compute_metrics_func(metric_names=['accuracy'], experiment_id=output_dir_suffix)
    
    # Initialize our Trainer
    trainer = Trainer(
        args=trainer_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
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