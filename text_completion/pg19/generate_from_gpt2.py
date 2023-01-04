

from torch.utils.data import DataLoader

from entropy_aware_search.hf_utils import DataArguments, ModelArguments, get_tokenizer, get_model
from utils import get_pg19_dataset, get_compute_metrics_func

import argparse
import logging

import numpy as np
import torch
import json
import os
import datetime
import timeit
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

def truncate(text):
    last_punc = 0
    if "." in text:
        last_punc = max(last_punc, text.rindex("."))
    if "?" in text:
        last_punc = max(last_punc, text.rindex("?"))
    if "!" in text:
        last_punc = max(last_punc, text.rindex("!"))
    if ";" in text:
        last_punc = max(last_punc, text.rindex(";"))
    if last_punc != 0:
        text = text[:last_punc + 1]
    return text

def printable_list(list, seperator=',', precision=2):
    return seperator.join([f"{round(x, precision)}" for x in list])

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
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--typical_p", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--entropy_aware_search", action="store_true", help="Use entropy aware search.")
    parser.add_argument("--ea_upper_limit_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_lower_limit_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_human_mean_coeffs", type=float, nargs='+', default=[-0.00277, 2.88702])
    parser.add_argument("--ea_human_std_coeffs", type=float, nargs='+', default=[-0.00064, 0.91427])
    parser.add_argument("--ea_version", type=int, default=3)
    parser.add_argument("--ea_patience_window", type=int, default=5)
    parser.add_argument("--ea_only_greedy_till", type=int, default=0)
    parser.add_argument('--ea_human_entropy_std_band', type=float, default=1)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
    )

    wiki_testset = get_pg19_dataset(
        "/home/mila/a/arorakus/wdir/entropy_aware_search/data/rankgen_data/")
        # split=['train[:10%]', 'test[:5%]'])


    tokenizer = get_tokenizer(model_args)
    tokenizer.pad_token = tokenizer.eos_token

    model = get_model(model_args)
    model = model.to(args.device)
    
    def tokenizer_method(examples):
        tokenized_examples = tokenizer(examples['prefix'], max_length=1024-128-3, truncation=True, padding=True, return_tensors='pt')
        return tokenized_examples, [x[0] for x in examples['targets']]

    # compute_metrics = get_compute_metrics_func(experiment_id="tmp_id", tokenizer=tokenizer, metric_names=['accuracy', 'mauve'])

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    model.config.top_k = None

    logger.info(args)

    compute_time = 0
    num_generations = 0
    with open(args.output_filename, 'w') as output_file:
        generated_output_sequences = []
        prompt_sequences = []
        targets = []
        for idx, batch_start in enumerate(range(0, len(wiki_testset), args.batch_size)):
            batch, batch_targets = tokenizer_method(wiki_testset[batch_start: batch_start+args.batch_size])
            batch = batch.to(args.device)
            start_time = timeit.default_timer()
            outputs = model.generate(
                **batch,
                max_new_tokens=128,
                min_length=20,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                typical_p=args.typical_p,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                entropy_aware_search=args.entropy_aware_search,
                return_dict_in_generate=True,
                # num_return_sequences=args.num_beams,
                output_scores=True,
                ea_version=args.ea_version,
                ea_lower_limit_coeffs=args.ea_lower_limit_coeffs,
                ea_upper_limit_coeffs=args.ea_upper_limit_coeffs,
                ea_patience_window=args.ea_patience_window,
                ea_only_greedy_till=args.ea_only_greedy_till,
                ea_human_mean_coeffs=args.ea_human_mean_coeffs,
                ea_human_std_coeffs=args.ea_human_std_coeffs,
                ea_human_std_band=args.ea_human_entropy_std_band,
            )
            end_time = timeit.default_timer()
            batch_size, batch_len = batch['input_ids'].shape
            batch_compute_time = int(end_time - start_time)
            compute_time += batch_compute_time
            num_generations += batch_size

            generated_outputs = outputs['sequences'][:, batch_len:]

            pct_entropy_violations = [-1] * batch_size
            pct_upper_entropy_violations = [-1] * batch_size
            pct_lower_entropy_violations = [-1] * batch_size

            entropies = [[]] * batch_size

            if args.entropy_aware_search:
                pct_entropy_violations =  outputs['pct_entropy_violations'].cpu().tolist()
                pct_upper_entropy_violations =  outputs['pct_upper_entropy_violations'].cpu().tolist()
                pct_lower_entropy_violations =  outputs['pct_lower_entropy_violations'].cpu().tolist()

                entropies = outputs['entropies'].cpu().tolist()

            input_ids = batch['input_ids']
            generated_output_sequences = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            prompt_sequences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            targets = batch_targets
        
            for generated_sequence_idx, (prompt_sequence, generated_sequence, target, 
                                            pct_violations, pct_upper_violations, pct_lower_violations, 
                                            seq_entropy) \
                    in enumerate(zip(prompt_sequences, generated_output_sequences, targets, pct_entropy_violations, pct_upper_entropy_violations, pct_lower_entropy_violations, 
                    entropies)):
                print(f"=== GENERATED SEQUENCE {idx}-{generated_sequence_idx + 1} ===", end='\r')
            
                generated_sequence = truncate(generated_sequence)
                target = target

                if (idx * args.batch_size + generated_sequence_idx) % 10 == 0:
                    print()
                    print('*' * 100)
                    print(f"Prompt: {prompt_sequence}")
                    print('-' * 100)
                    print(f"Generation: {generated_sequence}")
                    print('-' * 100)
                    print(f"Target: {target}")
                    print('*' * 100)
                    print()
                    if args.entropy_aware_search:
                        print(f"\tPercent violations: {pct_violations}")
                        print(f"\tPercent upper violations: {pct_upper_violations}")
                        print(f"\tPercent lower violations: {pct_lower_violations}")
                        print(f"\tCompute Time: {batch_compute_time/batch_size} secs")
                        # print(f"\tEntropies: {printable_list(seq_entropy, ' ', precision=1)}")
                    print('*' * 100)
                    print()

                output = {
                    'prefix': prompt_sequence, 
                    'generation': generated_sequence, 
                    'target': target,
                    'pct_violations': pct_violations,
                    'pct_upper_violations': pct_upper_violations,
                    'pct_lower_violations': pct_lower_violations,
                    'seq_mean_entropies': seq_entropy,
                    'compute_time': batch_compute_time/batch_size,
                }

                print(json.dumps(output), file=output_file, flush=True)


if __name__ == "__main__":
    main()
