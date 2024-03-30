

from datasets import load_dataset
from transformers import  AutoModelForCausalLM, AutoTokenizer

from accelerate import PartialState, Accelerator
from accelerate.utils import set_seed, gather_object

from tqdm import trange

import argparse
import logging

import torch
import json
import timeit
import os
import re

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

accelerator = Accelerator()
pattern = r"### Human: (.*?)### Assistant: (.*)"

def process_dataset(dataset_name, split):
    if dataset_name == 'tatsu-lab/alpaca_eval':
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split=split)
        dataset = dataset.rename_columns({
                    'instruction': 'prefix', 
                    'output': 'target'}).shuffle(42)
        prefix_key = 'instruction'
        response_key = 'output'
    elif dataset_name == 'timdettmers/openassistant-guanaco':
        dataset = load_dataset("timdettmers/openassistant-guanaco", split=split)
        transformed_dataset = []
        for d in dataset:
            text = d['text']
            matches = re.search(pattern, text,  re.DOTALL)
            transformed_dataset.append(
                {
                    "prefix": matches[1],
                    "target": matches[2],
                }
            )
            prefix_key = '### Human: '
            response_key = '### Assistant: '
        dataset = transformed_dataset

    return [x for x in dataset], prefix_key, response_key

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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="Mistral-7B-Instruct-v0.2",
        type=str,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        '--dataset_name',
        default="tatsu-lab/alpaca_eval",
        type=str, 
        help="Dataset to generate from"
    )
    parser.add_argument(
        '--split', 
        default='eval',
        type=str,
        help='Split to use for generation.',
    )
    parser.add_argument("--output_filename", type=str, default='/tmp/output.txt',  help="The output file to save the generation.")
    parser.add_argument("--length", type=int, default=3600)

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
    parser.add_argument("--do_sample", action="store_true", help="Use Sampling Decoding.")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--typical_p", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--max_prompt_length", default=2048, type=int, help="Maximum prompt length.")
    parser.add_argument("--max_length", default=4096, type=int, help="Maximum response length")
    parser.add_argument("--prompt_template", type=str, default="prompt_templates/default.template", help="Prompt template to use for evaluating models.")

    args = parser.parse_args()

    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    # logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.bf16}")
    distributed_state = PartialState(cpu=args.no_cuda)

    with open(args.prompt_template, 'r') as tf:
        template = tf.read()

    args.bf16 = True
    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    set_seed(42)

    if args.temperature is not None and args.temperature > 0:
        args.do_sample = True

    def tokenizer_method(examples):
        prompts = []
        prefixes = []
        targets = []
        for example in examples:
            prompt = template.replace("{instruction}", example['prefix']).replace("{input}", "")
            prompts.append(prompt)
            prefixes.append(example['prefix'])
            targets.append(example['target'])
        tokenized_examples = tokenizer(prompts, max_length=args.max_prompt_length, truncation=True, padding=True, return_tensors='pt')
        return tokenized_examples, prefixes,  targets

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
                                args.model_name_or_path,
                                trust_remote_code=True,
                                use_auth_token=True,
                                torch_dtype=torch_dtype,
                                low_cpu_mem_usage=True,
                                # attn_implementation="flash_attention_2",
                                )
    model.to(distributed_state.device)

    dataset, prefix_key, response_key = process_dataset(args.dataset_name, args.split)
    
    logger.info(args)

    compute_time = 0
    num_generations = 0
    output_jsons = []
    with open(args.output_filename, 'w') as output_file, \
        open(f"{args.output_filename}.metadata", 'w') as metadata_file, \
        distributed_state.split_between_processes(dataset) as dataset_split:
        generated_output_sequences = []
        print(f"{distributed_state.device} => {len(dataset_split)}")
        for idx, batch_start in enumerate(trange(0, len(dataset_split), args.batch_size)):
            batch, prefixes, targets = tokenizer_method(dataset_split[batch_start: batch_start+args.batch_size])

            batch = batch.to(distributed_state.device)

            start_time = timeit.default_timer()
            outputs = model.generate(
                **batch,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                typical_p=args.typical_p,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                return_dict_in_generate=True,
                output_scores=True,
            )

            end_time = timeit.default_timer()
            batch_size, batch_len = batch['input_ids'].shape

            batch_compute_time = int(end_time - start_time)
            compute_time += batch_compute_time
            num_generations += batch_size

            generated_outputs = outputs['sequences'][:, batch_len:]

            generated_output_sequences = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        
            for generated_sequence_idx, (prefix, generated_sequence, target) \
                    in enumerate(zip(prefixes, generated_output_sequences, targets)):
            
                # generated_sequence = truncate(generated_sequence)
                target = target

                if (idx * args.batch_size + generated_sequence_idx) % 10 == 0:
                    print()
                    print('*' * 100)
                    print(f"{distributed_state.device}")
                    print(f"Prompt: {prefix}")
                    print('-' * 100)
                    print(f"Generation: {generated_sequence}")
                    print('-' * 100)
                    print(f"Target: {target}")
                    print('*' * 100)
                    print()
                    print('*' * 100)
                    print()

                output = {
                    prefix_key: prefix, 
                    response_key: generated_sequence, 
                    'target': target,
                    'compute_time': batch_compute_time/batch_size,
                    'dataset': args.dataset_name,
                }

                output_jsons.append(output)
        all_output_jsons = gather_object(output_jsons)

        if PartialState().is_last_process:
            print(json.dumps(all_output_jsons, indent=2, sort_keys=True), file=output_file)
            opts = vars(args)
            opts['compute_time'] = compute_time
            opts['num_generations'] = num_generations
            opts['avg_compute_time'] = compute_time/num_generations

            print(json.dumps(opts, indent=2, sort_keys=True), 
                                file=metadata_file)
if __name__ == "__main__":
    main()
