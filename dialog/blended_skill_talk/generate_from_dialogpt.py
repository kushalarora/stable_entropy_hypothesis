

import json
import timeit

import argparse
import logging

import numpy as np
import torch


from datasets import load_dataset

from transformers import AutoTokenizer,AutoModelForCausalLM

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
    parser.add_argument("--output_filename", type=str, default="The output file to save the generation.", required=True)
    parser.add_argument(
        "--model_name_or_path",
        default="microsoft/DialoGPT-large",
        type=str,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument('--dataset', default='/home/mila/a/arorakus/wdir/entropy_aware_search/data/blended_skill_talk/generated/orig.jsonl')
    parser.add_argument("--length", type=int, default=128)
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
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--entropy_aware_search", action="store_true", help="Use entropy aware search.")
    parser.add_argument("--ea_upper_limit_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_lower_limit_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_human_mean_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_human_std_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_version", type=int, default=3)
    parser.add_argument("--ea_patience_window", type=int, default=5)
    parser.add_argument("--ea_only_greedy_till", type=int, default=5)
    parser.add_argument('--ea_human_entropy_std_band', type=float, default=1.0)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    dataset = load_dataset("json", data_files=args.dataset)
    
    dataset = dataset['train']
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)


    max_source_length =  args.length

    def tokenizer_function(examples):
        # remove pairs where at least one record is None
        model_inputs = tokenizer(examples['context'], 
                            max_length=max_source_length, 
                            truncation=True, padding=True, 
                            return_tensors='pt')
        return model_inputs, examples['model_text']



    if args.fp16:
        model.half()

    logger.info(args)

    compute_time = 0
    num_generations = 0
    with open(args.output_filename, 'w') as output_file:
        for idx, batch_start in enumerate(range(0, len(dataset), args.batch_size)):
            batch, batch_targets = tokenizer_function(dataset[batch_start: batch_start+args.batch_size])
            batch = batch.to(args.device)
            batch_size, batch_len = batch['input_ids'].shape

            start_time = timeit.default_timer()

            outputs = model.generate(
                **batch,
                max_new_tokens=128,
                min_length=batch_len+50,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                typical_p=args.typical_p,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=args.no_repeat_ngram_size,
                entropy_aware_search=args.entropy_aware_search,
                # return_dict_in_generate=True,
                # output_scores=True,
                # version=args.version,
                # lower_limit_coeffs=args.ea_lower_limit_coeffs,
                # upper_limit_coeffs=args.ea_upper_limit_coeffs,
                # patience_window=args.patience_window,
                # only_greedy_till=args.only_greedy_till,
                # human_mean_coeffs=args.ea_human_mean_coeffs,
                # human_std_coeffs=args.ea_human_std_coeffs,
                # human_std_band=args.ea_human_entropy_std_band,
            )
            end_time = timeit.default_timer()
            batch_compute_time = int(end_time - start_time)
            compute_time += batch_compute_time
            num_generations += batch_size

            generated_outputs = outputs[:, batch_len:]

            generated_responses = tokenizer.batch_decode(generated_outputs,             
                                                            skip_special_tokens=True)
            contexts = tokenizer.batch_decode(batch['input_ids'],  skip_special_tokens=True)
            targets = batch_targets

            pct_entropy_violations = [-1] * batch_size
            pct_upper_entropy_violations = [-1] * batch_size
            pct_lower_entropy_violations = [-1] * batch_size

            entropies = [[]] * batch_size

            if args.entropy_aware_search:
                pct_entropy_violations =  outputs['pct_entropy_violations'].cpu().tolist()
                pct_upper_entropy_violations =  outputs['pct_upper_entropy_violations'].cpu().tolist()
                pct_lower_entropy_violations =  outputs['pct_lower_entropy_violations'].cpu().tolist()

                entropies = outputs['entropies'].cpu().tolist()


            for generated_sequence_idx, (context, generated_response, target, pct_violations, pct_upper_violations, pct_lower_violations, seq_entropy) \
                    in enumerate(zip(contexts, generated_responses, targets, pct_entropy_violations, pct_upper_entropy_violations, pct_lower_entropy_violations, entropies)):
                print(f"=== GENERATED SEQUENCE {idx}-{generated_sequence_idx + 1} ===", end='\r')
            
                if (idx * args.batch_size + generated_sequence_idx) % 100 == 0:
                    print()
                    print('*' * 100)
                    print(context)
                    print('-' * 100)
                    print(generated_response)
                    print('-' * 100)
                    print(target)
                    print('*' * 100)
                    print()
                    if args.entropy_aware_search:
                        print(f"\tPercent violations: {pct_violations}")
                        print(f"\tPercent upper violations: {pct_upper_violations}")
                        print(f"\tPercent lower violations: {pct_lower_violations}")
                        print(f"\tCompute Time: {batch_compute_time/batch_size} secs")
                    print('*' * 100)
                    print()

                output = {
                    'prefix': context, 
                    'generation': generated_response, 
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
