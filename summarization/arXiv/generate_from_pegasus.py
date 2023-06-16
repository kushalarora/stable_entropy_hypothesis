

import json
import timeit
from torch.utils.data import DataLoader

from entropy_aware_search.hf_utils import ModelArguments, get_tokenizer, get_model

import argparse
import logging

import numpy as np
import torch
import evaluate

from datetime import datetime

from datasets import load_dataset

from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
experiment_id = "pegasus_bigbird_arxiv_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

rouge = evaluate.load("rouge", experiment_id=experiment_id, 
                        cache_dir=f"/tmp/{experiment_id}.txt")

import numpy as np

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
    parser.add_argument("--max_source_length", type=int, default=4096)
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
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--typical_p", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--entropy_aware_search", action="store_true", help="Use entropy aware search.")
    parser.add_argument("--ea_human_mean_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_human_std_coeffs", type=float, nargs='+')
    parser.add_argument("--ea_version", type=int, default=4)
    parser.add_argument("--ea_patience_window", type=int, default=5)
    parser.add_argument('--ea_human_entropy_std_band', type=float, default=1.0)
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

    args.max_source_length = adjust_length_to_model(args.max_source_length, max_sequence_length=model.config.max_position_embeddings)

    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    text_column = "article"
    summary_column = "abstract"
    max_source_length =  args.max_source_length

    def tokenizer_method(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding=True, return_tensors='pt')
        return model_inputs, targets

    if args.fp16:
        model.half()

    logger.info(args)
    arxiv_summ_testset = arxiv_summ_dataset['test']


    compute_time = 0
    num_generations = 0
    generated_sequences = []
    with open(args.output_filename, 'w') as output_file, \
        open(f"{args.output_filename}.rouge", 'w') as rouge_score_file:
        rouges_dict = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
        for idx, batch_start in enumerate(range(0, len(arxiv_summ_testset), args.batch_size)):
            batch, targets = tokenizer_method(arxiv_summ_testset[batch_start: batch_start+args.batch_size])            
            batch = batch.to(args.device)
            start_time = timeit.default_timer()
            outputs = model.generate(
                **batch,
                max_length=256,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                typical_p=args.typical_p,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                entropy_aware_search=args.entropy_aware_search,
                return_dict_in_generate=True,
                output_scores=True,
                ea_version=args.ea_version,
                ea_patience_window=args.ea_patience_window,
                ea_human_mean_coeffs=args.ea_human_mean_coeffs,
                ea_human_std_coeffs=args.ea_human_std_coeffs,
                ea_human_std_band=args.ea_human_entropy_std_band,
            )
            end_time = timeit.default_timer()
            batch_size, batch_len = batch['input_ids'].shape
            batch_compute_time = int(end_time - start_time)
            compute_time += batch_compute_time
            num_generations += batch_size

            generated_abstracts = tokenizer.batch_decode(outputs['sequences'],  skip_special_tokens=True)
            articles = tokenizer.batch_decode(batch['input_ids'],  skip_special_tokens=True)

            rouge_scores = rouge.compute(predictions=generated_abstracts, 
                                        references=targets, use_stemmer=True)

            for key,score in rouge_scores.items():
                rouges_dict[key].append(score)

            pct_entropy_violations = [-1] * batch_size
            pct_upper_entropy_violations = [-1] * batch_size
            pct_lower_entropy_violations = [-1] * batch_size
            entropies = [[]] * batch_size

            if args.entropy_aware_search:
                pct_entropy_violations =  outputs['pct_entropy_violations'].cpu().tolist()
                pct_upper_entropy_violations =  outputs['pct_upper_entropy_violations'].cpu().tolist()
                pct_lower_entropy_violations =  outputs['pct_lower_entropy_violations'].cpu().tolist()

                entropies = outputs['entropies'].cpu().tolist()

            for generated_sequence_idx, \
                (article, generated_abstract, target, 
                    pct_violations, pct_upper_violations, 
                    pct_lower_violations, seq_entropy) \
                    in enumerate(zip(articles, generated_abstracts, targets, 
                        pct_entropy_violations, pct_upper_entropy_violations, 
                        pct_lower_entropy_violations, entropies)):
                print(f"=== GENERATED SEQUENCE {idx}-{generated_sequence_idx + 1} ===", end='\r')
            
                article = article.strip()
                generated_abstract = generated_abstract.strip()

                if (idx * args.batch_size + generated_sequence_idx) % 100 == 0:
                    print()
                    print('*' * 100)
                    print(f"Article: \n{article}")
                    print('-' * 100)
                    print(f"Generated Abstract: \n {generated_abstract}")
                    print('-' * 100)
                    print(f"Original Abstract: \n {target}")
                    print('*' * 100)

                    if args.entropy_aware_search:
                        print(f"\tPercent violations: {pct_violations}")
                        print(f"\tPercent upper violations: {pct_upper_violations}")
                        print(f"\tPercent lower violations: {pct_lower_violations}")
                        print(f"\tCompute Time: {batch_compute_time/batch_size} secs")
                    print('*' * 100)
                    print()

                output = {
                    'prefix': article, 
                    'generation': generated_abstract, 
                    'target': target,
                    'pct_violations': pct_violations,
                    'pct_upper_violations': pct_upper_violations,
                    'pct_lower_violations': pct_lower_violations,
                    'seq_mean_entropies': seq_entropy,
                    'compute_time': batch_compute_time/batch_size,
                }

                print(json.dumps(output), file=output_file, flush=True)
        
        for key,scores in rouges_dict.items():
            rouges_dict[key] = np.mean(scores)

        print(json.dumps(rouges_dict, indent=True, sort_keys=True))
        print(json.dumps(rouges_dict, indent=True, sort_keys=True), file=rouge_score_file, flush=True)
if __name__ == "__main__":
    main()