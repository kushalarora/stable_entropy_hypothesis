
import sys
import os

from transformers import default_data_collator
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))
from entropy_beam_search.hf_utils import DataArguments, ModelArguments, get_dataset, get_model, get_prompts, get_tokenizer, group_into_blocks, tokenize

from torch.utils.data import DataLoader

per_device_batch_size = 8

data_args = DataArguments(
    dataset_name = "rewardsignal/reddit_writing_prompts",
    data_files="prompt_responses_full.csv",
)


model_args = ModelArguments(
    model_name_or_path="gpt2-large"
)

dataset = get_dataset(data_args)


tokenizer = get_tokenizer(model_args)
# tokenizer = lambda txt_lst: [txt.split() for txt in txt_lst]

tokenized_dataset = tokenize(dataset, tokenizer)
lm_datasets = group_into_blocks(tokenized_dataset, block_size=512)

prompt_dataset = get_prompts(lm_datasets, prompt_len=50)

model = get_model(model_args)

train_dataset = prompt_dataset["train"]
eval_dataset = prompt_dataset["validation"]
test_dataset = prompt_dataset["test"]

# DataLoaders creation:
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=per_device_batch_size
)
eval_dataloader = DataLoader(
    eval_dataset, collate_fn=default_data_collator, batch_size=per_device_batch_size)


for step, batch in enumerate(train_dataloader):
    model.generate()
