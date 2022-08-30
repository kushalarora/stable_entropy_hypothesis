

from torch.utils.data import DataLoader

from entropy_aware_search.hf_utils import DataArguments, ModelArguments, get_tokenizer, get_model
from utils import get_writing_prompt_dataset, preprocess_logits_for_metrics, get_tokenized_prompt_dataset, get_compute_metrics_func

import argparse
import logging

import numpy as np
import torch

from transformers import (
    DataCollatorForLanguageModeling,
 
)


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

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )

    parser.add_argument("--output_filename", type=str, default="The output file to save the generation.")
    parser.add_argument("--length", type=int, default=1024)
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
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
    )

    prompt_response_dataset = get_writing_prompt_dataset("/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/")

    tokenizer = get_tokenizer(model_args)
    tokenizer.pad_token = tokenizer.eos_token

    model = get_model(model_args)
    model = model.to(args.device)

    tokenized_writing_prompt_dataset = get_tokenized_prompt_dataset(prompt_response_dataset, tokenizer, generate=True)
    tokenized_writing_prompt_testset = tokenized_writing_prompt_dataset['test']

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    # DataLoaders creation:
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    test_dataloader = DataLoader(tokenized_writing_prompt_testset, collate_fn=data_collator, batch_size=args.batch_size)

    generated_sequences = []
    with open(args.output_filename, 'w') as output_file:

        for idx, batch in enumerate(test_dataloader):
            batch = batch.to(args.device)
            outputs = model.generate(
                **batch,
                max_length=args.length,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                typical_p=args.typical_p,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                # num_return_sequences=args.num_beams,
            )

            batch_len = batch['input_ids'].shape[1]
            generated_outputs = outputs[:, batch_len:]

            generated_output_sequences = tokenizer.batch_decode(generated_outputs)
            prompt_sequences = tokenizer.batch_decode(batch['input_ids'])

            for generated_sequence_idx, (prompt_sequence, generated_sequence) \
                    in enumerate(zip(prompt_sequences, generated_output_sequences)):
                print(f"=== GENERATED SEQUENCE {idx}-{generated_sequence_idx + 1} ===", end='\r')
            
                # Remove all text after the stop token
                prompt_sequence = prompt_sequence[:prompt_sequence.find(args.stop_token) if args.stop_token else None]
                generated_sequence = generated_sequence[:generated_sequence.find(args.stop_token) if args.stop_token else None]

                prompt_sequence = prompt_sequence.strip().replace("\n", "<newline>")
                generated_sequence = generated_sequence.strip().replace("\n", "<newline>")

                generated_sequences.append((prompt_sequence, generated_sequence))

                if (idx * args.batch_size + generated_sequence_idx) % 100 == 0:
                    print()
                    print('*' * 100)
                    print(prompt_sequence)
                    print(generated_sequence)
                    print('*' * 100)

                print(f"{prompt_sequence}\t{generated_sequence}", file=output_file)


if __name__ == "__main__":
    main()
