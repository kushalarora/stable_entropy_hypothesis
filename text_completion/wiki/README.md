
### Generate from GPT-2 XL (Wiki dataset RankGen)
```bash
sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/greedy.jsonl --fp16

sbatch -t 48:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/beam_5.jsonl --num_beams 5 --batch_size 4 --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/top_p_0.9.jsonl  --p 0.9 --do_sample --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/top_p_0.95.jsonl  --p 0.95 --do_sample --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/top_k_30.jsonl  --k 30 --do_sample --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/top_k_40.jsonl  --k 40 --do_sample --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/typical_p_0.9.jsonl  --typical_p 0.9 --do_sample --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/typical_p_0.2.jsonl  --typical_p 0.2 --do_sample --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/temp_1.jsonl  --temperature 1 --do_sample --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/temp_1_2.jsonl  --temperature 1.2 --do_sample --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/temp_0_8.jsonl  --temperature 0.8 --do_sample --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/temp_0_5.jsonl  --temperature 0.5 --do_sample --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/eags_v2_k_30.jsonl --k 30 --eags_version 2 --fp16 --entropy_aware_search

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/eags_v1_k_30.jsonl --k 30 --eags_version 1 --fp16 --entropy_aware_search

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/eags_v3_k_30.jsonl --k 30 --eags_version 3 --fp16 --entropy_aware_search

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/eags_v3_temp_1.jsonl --temperature 1 --eags_version 3 --fp16 --entropy_aware_search

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/eags_v3_typical_0.2.jsonl --typical_p 0.2 --version 3 --fp16 --entropy_aware_search --ea_upper_limit_coeffs -0.0 0.00033 -0.01698 5.5549 --ea_lower_limit_coeffs -0.0 1e-05 0.00102 1.43047

sbatch -t 10:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/eags_v4_typical_0.2.jsonl --typical_p 0.2 --version 4 --fp16 --entropy_aware_search --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --batch_size 8
```