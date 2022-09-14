from torch.utils.data import DataLoader

from utils import get_writing_prompt_dataset, preprocess_logits_for_metrics, get_tokenized_prompt_dataset, compute_metrics



writing_prompt_dataset = get_writing_prompt_dataset("/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/")

with open('/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/generated/orig.txt', 'w') as orig_file:
    for datapoint in writing_prompt_dataset['test']:
        prompt_sequence = datapoint['prompt'].strip().replace("\n", "<newline>").replace("\t", "<tab>")
        generated_sequence = datapoint['response'].strip().replace("\n", "<newline>").replace("\t", "<tab>")
        print(f"{prompt_sequence}\t{generated_sequence}", file=orig_file)