
### Generate from GPT-2 XL (pg19 dataset RankGen)
```bash
sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/greedy.jsonl --fp16

sbatch -t 48:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/beam_5.jsonl --num_beams 5 --batch_size 4 --fp16

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/top_p_0.9.csv  --p 0.9 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/top_p_0.95.csv  --p 0.95 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/top_k_30.csv  --k 30 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/top_k_40.csv  --k 40 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/typical_p_0.95.csv  --typical_p 0.95 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/typical_p_0.2.csv  --typical_p 0.2 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/temp_1.csv  --temperature 1 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/temp_1_2.csv  --temperature 1.2 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/pg19/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/pg19_rankgen/generated/gpt2_xl/temp_0_8.csv  --temperature 0.8 --do_sample

```