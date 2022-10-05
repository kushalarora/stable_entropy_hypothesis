!#/bin/bash

for temp in 0.8 0.9 1 1.1; do
    sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/v2_sampling_analysis/eags_v2_temp_${temp}.jsonl --temperature ${temp} --eags_version 2 --fp16 --entropy_aware_search
done

for k in 10 30 50 100; do
    sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/v2_sampling_analysis/eags_v2_k_${k}.jsonl --k ${k} --eags_version 2 --fp16 --entropy_aware_search
done

for tau in 0.2 0.5 0.9 0.95; do
    sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/v2_sampling_analysis/eags_v2_typical_${tau}.jsonl --typical_p ${tau} --eags_version 2 --entropy_aware_search --batch_size 16
done


for p in 0.2 0.5 0.9 0.95; do
    sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/v2_sampling_analysis/eags_v2_p_${p}.jsonl --p ${p} --eags_version 2 --fp16 --entropy_aware_search
done
