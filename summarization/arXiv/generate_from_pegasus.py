

from torch.utils.data import DataLoader

from entropy_aware_search.hf_utils import ModelArguments, get_tokenizer, get_model

import argparse
import logging

import numpy as np
import torch

from transformers import (
    DataCollatorForSeq2Seq,
)

from datasets import load_dataset

from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_filename", type=str, default="The output file to save the generation.")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--stop_token", type=str, default="<|endoftext|>", help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--p", type=float, default=None)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--do_sample", action="store_true", help="Use Sampling Decoding.")
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--typical_p", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--entropy_aware_search", action="store_true", help="Use entropy aware search.")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    model_args = ModelArguments(
        model_name_or_path="google/bigbird-pegasus-large-arxiv"
    )

    arxiv_summ_dataset = load_dataset("scientific_papers", "arxiv")
  
    model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")

    model = model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)

    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")


    text_column = "article"
    summary_column = "abstract"
    max_source_length =  args.length

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding=False)
        return model_inputs

    arxiv_summ_testset = arxiv_summ_dataset['test']

    tokenized_arxiv_summ_testset = arxiv_summ_testset.map(
        preprocess_function,
        batched=True,
        num_proc=10,
        remove_columns=arxiv_summ_dataset['test'].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on test dataset",
    )

    if args.fp16:
        model.half()

    logger.info(args)

    # DataLoaders creation:
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,)
    test_dataloader = DataLoader(tokenized_arxiv_summ_testset, collate_fn=data_collator, batch_size=args.batch_size)


    with open(args.output_filename, 'w') as output_file:

        for idx, batch in enumerate(test_dataloader):
            batch = batch.to(args.device)
            outputs = model.generate(
                **batch,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                typical_p=args.typical_p,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                entropy_aware_search=args.entropy_aware_search,
            )
            generated_abstracts = tokenizer.batch_decode(outputs,  skip_special_tokens=True)
            articles = tokenizer.batch_decode(batch['input_ids'],  skip_special_tokens=True)

            for generated_sequence_idx, (article, generated_abstract) \
                    in enumerate(zip(articles, generated_abstracts)):
                print(f"=== GENERATED SEQUENCE {idx}-{generated_sequence_idx + 1} ===", end='\r')
            
               
                article = article.strip().replace("\n", "<newline>")
                generated_abstract = generated_abstract.strip().replace("\n", "<newline>")

                if (idx * args.batch_size + generated_sequence_idx) % 100 == 0:
                    print()
                    print('*' * 100)
                    print(article)
                    print(generated_abstract)
                    print('*' * 100)

                print(f"{article}\t{generated_abstract}", file=output_file, flush=True)


if __name__ == "__main__":
    main()
