sbatch -t 30:00 ./launcher_basic.sh python text_completion/factuality/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/factuality/generated/gpt2_xl/greedy_factual.jsonl --fp16 --batch_size 128 --type factual 

sbatch -t 30:00 ./launcher_basic.sh python text_completion/factuality/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/factuality/generated/gpt2_xl/greedy_nonfactual.jsonl --fp16 --batch_size 128 --type nonfactual 

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/factuality/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/factuality/generated/gpt2_xl/beam_5_factual.jsonl --num_beams 5 --batch_size 16 --fp16 --type factual

sbatch -t 2:00:00 ./launcher_basic.sh python text_completion/factuality/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/factuality/generated/gpt2_xl/beam_5_nonfactual.jsonl --num_beams 5 --batch_size 16 --fp16 --type nonfactual

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/factuality/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/factuality/generated/gpt2_xl/typical_p_0.2_factual.jsonl  --typical_p 0.2 --do_sample --fp16 --batch_size 128 --type factual 

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/factuality/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/factuality/generated/gpt2_xl/typical_p_0.2_nonfactual.jsonl  --typical_p 0.2 --do_sample --fp16 --batch_size 128 --type nonfactual 


sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/factuality/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/factuality/generated/gpt2_xl/top_p_0.9_factual.jsonl  --p 0.9 --do_sample --fp16 --batch_size 128 --type factual 

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/factuality/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/factuality/generated/gpt2_xl/top_p_0.9_nonfactual.jsonl  --p 0.9 --do_sample --fp16 --batch_size 128 --type nonfactual 

sbatch -t 8:00:00 ./launcher_basic.sh python text_completion/factuality/generate_from_gpt2.py --model_name_or_path gpt2-xl --typical_p 0.2 --ea_version 4 --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --ea_human_entropy_std_band 0.8 --batch_size 4 --ea_patience_window 5 --ea_only_greedy_till 5 --output_filename data/factuality/generated/gpt2_xl/eags_v4_tau_0.2_std_dev_0.8_till_5_patience_5_factual.jsonl --fp16 --type factual


sbatch -t 8:00:00 ./launcher_basic.sh python text_completion/factuality/generate_from_gpt2.py --model_name_or_path gpt2-xl --typical_p 0.2 --ea_version 4 --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --ea_human_entropy_std_band 0.8 --batch_size 4 --ea_patience_window 5 --ea_only_greedy_till 5 --output_filename data/factuality/generated/gpt2_xl/eags_v4_tau_0.2_std_dev_0.8_till_5_patience_5_nonfactual.jsonl --fp16 --type nonfactual



PYTHONPATH=~/wdir/entropy_aware_search/FactualityPrompt python src/evaluate_v3_final.py --prompt_type factual --gen_path data/factuality/generated/gpt2_xl/greedy_factual.jsonl

PYTHONPATH=~/wdir/entropy_aware_search/FactualityPrompt python src/evaluate_v3_final.py --prompt_type nonfactual --gen_path data/factuality/generated/gpt2_xl/greedy_nonfactual.jsonl

PYTHONPATH=~/wdir/entropy_aware_search/FactualityPrompt python src/evaluate_v3_final.py --prompt_type factual --gen_path data/factuality/generated/gpt2_xl/beam_5_factual.jsonl

PYTHONPATH=~/wdir/entropy_aware_search/FactualityPrompt python src/evaluate_v3_final.py --prompt_type nonfactual --gen_path data/factuality/generated/gpt2_xl/beam_5_nonfactual.jsonl


PYTHONPATH=~/wdir/entropy_aware_search/FactualityPrompt python src/evaluate_v3_final.py --prompt_type factual --gen_path data/factuality/generated/gpt2_xl/typical_p_0.2_factual.jsonl

PYTHONPATH=~/wdir/entropy_aware_search/FactualityPrompt python src/evaluate_v3_final.py --prompt_type nonfactual --gen_path data/factuality/generated/gpt2_xl/typical_p_0.2_nonfactual.jsonl

PYTHONPATH=~/wdir/entropy_aware_search/FactualityPrompt python src/evaluate_v3_final.py --prompt_type factual --gen_path data/factuality/generated/gpt2_xl/top_p_0.9_factual.jsonl

PYTHONPATH=~/wdir/entropy_aware_search/FactualityPrompt python src/evaluate_v3_final.py --prompt_type nonfactual --gen_path data/factuality/generated/gpt2_xl/top_p_0.9_nonfactual.jsonl
