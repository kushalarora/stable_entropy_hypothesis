

from torch.utils.data import DataLoader
from transformers import  AutoModelForCausalLM, AutoTokenizer

from accelerate import PartialState, Accelerator
from accelerate.utils import set_seed, gather_object

# from entropy_aware_search.hf_utils import DataArguments, ModelArguments, get_tokenizer, get_model
from tqdm import trange
from utils import get_wiki_dataset, get_compute_metrics_func

import argparse
import logging

import os
import numpy as np
import torch
import json
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
    parser.add_argument("--length", type=int, default=512)

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
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
        "--bf16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--do_sample", action="store_true", help="Use Sampling Decoding.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--typical_p", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--entropy_aware_sampling", action="store_true", help="Use entropy aware search.")
    parser.add_argument("--ea_upper_limit_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_lower_limit_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_human_mean_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_human_std_coeffs", type=float, nargs='+')
    # parser.add_argument("--ea_human_mean_coeffs", type=float, nargs='+', default=[-0.00277, 2.88702])
    # parser.add_argument("--ea_human_std_coeffs", type=float, nargs='+', default=[-0.00064, 0.91427])
    parser.add_argument("--ea_version", type=int, default=3)
    parser.add_argument("--ea_patience_window", type=int, default=5)
    parser.add_argument("--ea_only_greedy_till", type=int, default=5)
    parser.add_argument('--ea_human_entropy_std_band', type=float, default=1.0)
    parser.add_argument("--ea_donot_intervene_for_lower_bound", action="store_true", help="Use Sampling Decoding.")
    parser.add_argument("--ea_donot_intervene_for_upper_bound", action="store_true", help="Use Sampling Decoding.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load in 8 bit.")
    parser.add_argument("--num_examples", type=int, default=-1)

    args = parser.parse_args()

    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    distributed_state = PartialState(cpu=args.no_cuda)
    set_seed(args)

    wiki_testset = get_wiki_dataset(
        os.path.expanduser("~/wdir/stable_entropy_hypothesis/data/text_completion/"))
        # split=['train[:10%]', 'test[:5%]'])
    
    wiki_testset = wiki_testset[:args.num_examples]
    args.bf16 = True
    torch_dtype = None
    if args.bf16:
        if args.load_in_8bit:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16

    if args.temperature is not None and args.temperature > 0:
        args.do_sample = True

    def tokenizer_method(examples):
        prefixes = []
        targets = []
        for example in examples:
            prefixes.append(example['context'])
            targets.append(example['model_text'])

        tokenized_examples = tokenizer(prefixes, max_length=512, truncation=True, padding=True, return_tensors='pt')
        return tokenized_examples, prefixes, targets

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
                                args.model_name_or_path,
                                trust_remote_code=True,
                                use_auth_token=True,
                                torch_dtype=torch_dtype,
                                low_cpu_mem_usage=True,
                                load_in_8bit=args.load_in_8bit,
                                attn_implementation="flash_attention_2",
                                device_map={"": distributed_state.device}
                                )


    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    model.config.top_k = None

    logger.info(args)


    compute_time = 0
    num_generations = 0
    output_jsons = []
    with open(args.output_filename, 'w') as output_file, \
        open(f"{args.output_filename}.metadata", 'w') as metadata_file, \
        distributed_state.split_between_processes(wiki_testset) as dataset_split:
        generated_output_sequences = []
        prompt_sequences = []
        targets = []
        print(f"{distributed_state.device} => {len(dataset_split)}")
        for idx, batch_start in enumerate(trange(0, len(dataset_split), args.batch_size)):
            batch, prefixes, batch_targets = tokenizer_method(dataset_split[batch_start: batch_start+args.batch_size])
            batch = batch.to(distributed_state.device)
            start_time = timeit.default_timer()
            outputs = model.generate(
                **batch,
                max_new_tokens=args.length,
                min_length=20,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                typical_p=args.typical_p,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                # entropy_aware_sampling=args.entropy_aware_sampling,
                return_dict_in_generate=True,
                output_scores=True,
                # ea_version=args.ea_version,
                # ea_lower_limit_coeffs=args.ea_lower_limit_coeffs,
                # ea_upper_limit_coeffs=args.ea_upper_limit_coeffs,
                # ea_patience_window=args.ea_patience_window,
                # ea_only_greedy_till=args.ea_only_greedy_till,
                # entropy_aware_human_mean_coeffs=args.ea_human_mean_coeffs,
                # entropy_aware_human_std_coeffs=args.ea_human_std_coeffs,
                # entropy_aware_human_std_band=args.ea_human_entropy_std_band,
                # pad_token_id=tokenizer.eos_token_id,
                # ea_intervene_for_lower_bound=not args.ea_donot_intervene_for_lower_bound,
                # ea_intervene_for_upper_bound=not args.ea_donot_intervene_for_upper_bound,
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
            num_backoffs = 0
            entropies = [[]] * batch_size

            entropy_aware_search = args.ea_human_mean_coeffs is not None and \
                                    args.ea_human_std_coeffs is not None and \
                                        not args.entropy_aware_sampling
            if entropy_aware_search:
                pct_entropy_violations =  outputs['pct_entropy_violations'].cpu().tolist()
                pct_upper_entropy_violations =  outputs['pct_upper_entropy_violations'].cpu().tolist()
                pct_lower_entropy_violations =  outputs['pct_lower_entropy_violations'].cpu().tolist()
                backoff_indexes = outputs['backoff_indexes']
                num_backoffs =  len(backoff_indexes)
                entropies = outputs['entropies'].cpu().tolist()

            input_ids = batch['input_ids']
            generated_output_sequences = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            prompt_sequences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            targets = batch_targets
        
            for generated_sequence_idx, (prompt_sequence, generated_sequence, target, 
                                            pct_violations, pct_upper_violations, pct_lower_violations, 
                                            seq_entropy) \
                    in enumerate(zip(prompt_sequences, generated_output_sequences, targets, 
                        pct_entropy_violations, pct_upper_entropy_violations, pct_lower_entropy_violations, 
                        entropies)):
                # print(f"=== GENERATED SEQUENCE {idx}-{generated_sequence_idx + 1} ===", end='\r')
            
                generated_sequence = truncate(generated_sequence)
                target = target

                if (idx * args.batch_size + generated_sequence_idx) % 8 == 0:
                    print()
                    print('*' * 100)
                    print(f"{distributed_state.device}")
                    print(f"Prompt: {prompt_sequence}")
                    print('-' * 100)
                    print(f"Generation: {generated_sequence}")
                    print('-' * 100)
                    print(f"Target: {target}")
                    print('*' * 100)
                    print()
                    if entropy_aware_search:
                        print(f"\tPercent violations: {pct_violations}")
                        print(f"\tPercent upper violations: {pct_upper_violations}")
                        print(f"\tPercent lower violations: {pct_lower_violations}")
                        print(f"\tBackoff Indexes: {backoff_indexes}")
                        print(f"\tNum Backoff: {num_backoffs}")
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
                    'num_backoffs': num_backoffs,
                    'compute_time': batch_compute_time/batch_size,
                }
                output_jsons.append(output)
        all_output_jsons = gather_object(output_jsons)

        if PartialState().is_last_process:
            print(json.dumps(all_output_jsons, indent=2, sort_keys=True), file=output_file)

        opts = vars(args)
        opts['compute_time'] = compute_time
        opts['num_generations'] = num_generations
        opts['avg_compute_time'] = compute_time/num_generations

        print(json.dumps(output, indent=2, sort_keys=True), 
                                file=metadata_file, flush=True)
if __name__ == "__main__":
    main()
