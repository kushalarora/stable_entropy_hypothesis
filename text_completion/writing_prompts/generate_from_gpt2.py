

from torch.utils.data import DataLoader

from entropy_aware_search.hf_utils import DataArguments, ModelArguments, get_tokenizer, get_model
from utils import get_writing_prompt_dataset, get_tokenized_prompt_dataset, get_compute_metrics_func

import argparse
import logging

import numpy as np
import torch

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--entropy_aware_search", action="store_true", help="Use entropy aware search.")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
    )

    prompt_response_dataset = get_writing_prompt_dataset(
        "/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/")
        # split=['train[:10%]', 'test[:5%]'])

    # prompt_response_dataset = filter_out_shorter_prompt(prompt_response_dataset)

    tokenizer = get_tokenizer(model_args)
    tokenizer.pad_token = tokenizer.eos_token

    model = get_model(model_args)
    model = model.to(args.device)
    prompt_response_testset = prompt_response_dataset['test']
    
    def tokenizer_method(examples):
        contexts = []
        gold_responses = []
        for prompt,response in zip(examples['prompt'], examples['response']):
            response_tok = response.strip().split()
            context_toks = prompt.split()
            response = ' '.join(response_tok[128 - len(context_toks):])
            context_toks += ['<newline>'] + response_tok[:128 -1 - len(context_toks)]

            assert len(context_toks) == 128

            context = ' '.join(context_toks)

            contexts.append(context)
            gold_responses.append(response)

        tokenized_examples = tokenizer(contexts, max_length=768, truncation=True, padding=True, return_tensors='pt')

        return tokenized_examples, gold_responses

    # tokenized_writing_prompt_dataset = prompt_response_dataset.map(
    #                     tokenizer_method, 
    #                     batched=True, 
    #                     num_proc=10, 
    #                     # load_from_cache_file=True,
    #                     remove_columns=prompt_response_dataset['validation'].column_names)

    # tokenized_writing_prompt_dataset = get_tokenized_prompt_dataset(prompt_response_dataset, tokenizer, generate=True)
    # tokenized_writing_prompt_testset = tokenized_writing_prompt_dataset['test']

    # compute_metrics = get_compute_metrics_func(experiment_id="tmp_id", tokenizer=tokenizer, metric_names=['accuracy', 'mauve'])

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    model.config.top_k = None

    logger.info(args)

    # DataLoaders creation:
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # test_dataloader = DataLoader(tokenized_writing_prompt_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    
    generated_sequences = []
    with open(args.output_filename, 'w') as output_file:
        generated_output_sequences = []
        prompt_sequences = []
        targets = []
        for idx, batch_start in enumerate(range(0, len(prompt_response_testset), args.batch_size)):
            batch, batch_targets = tokenizer_method(prompt_response_testset[batch_start: batch_start+args.batch_size])
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
                entropy_aware_search=args.entropy_aware_search,
                return_dict_in_generate=True,
                num_return_sequences=args.num_beams,
                output_scores=True,
            )

            batch_size, batch_len = batch['input_ids'].shape
            generated_outputs = outputs['sequences'][:, batch_len:]

            pct_entropy_voilations = [-1] * batch_size
            entropies = [[]] * batch_size

            if args.entropy_aware_search:
                pct_entropy_voilations =  outputs['pct_entropy_voilations'].cpu().tolist()
                entropies = outputs['entropies'].cpu().tolist()

            input_ids = batch['input_ids']
            generated_output_sequences = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            prompt_sequences = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            targets = batch_targets
        
            for generated_sequence_idx, (prompt_sequence, generated_sequence, target, seq_pct_voilations, seq_entropy) \
                    in enumerate(zip(prompt_sequences, generated_output_sequences, targets, pct_entropy_voilations, entropies)):
                print(f"=== GENERATED SEQUENCE {idx}-{generated_sequence_idx + 1} ===", end='\r')
            
                prompt_sequence = prompt_sequence.strip().replace("\n", "<newline>")
                generated_sequence = generated_sequence.strip().replace("\n", "<newline>")
                target = target.strip().replace("\n", "<newline>")

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
                        print(f"\tPercent Voilations: {seq_pct_voilations}")
                        print(f"\tEntropies: {printable_list(seq_entropy, ' ', precision=1)}")
                    print('*' * 100)
                    print()


                print(f"{prompt_sequence}\t{generated_sequence}\t{target}\t"+
                        f"{seq_pct_voilations}\t{printable_list(seq_entropy, precision=3)}", 
                        file=output_file, flush=True)

if __name__ == "__main__":
    main()