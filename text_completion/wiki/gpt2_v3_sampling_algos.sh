!#/bin/bash
#### DEPRECATED #####
for std_dev in 0.8 1.0 1.2 1.5; do
    for temp in 0.8 0.9 1 1.1; do
        sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/v3_sampling_analysis/eags_v3_temp_${temp}.jsonl --temperature ${temp} --ea_version 4 --fp16 --entropy_aware_search 
    done

    for k in 10 30 50 100; do
        sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/v3_sampling_analysis/eags_v3_k_${k}.jsonl --k ${k} --ea_version 4 --fp16 --entropy_aware_search
    done

    for tau in 0.2 0.5 0.9 0.95; do
        sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/v3_sampling_analysis/eags_v3_typical_${tau}.jsonl --typical_p ${tau} --ea_version 4 --entropy_aware_search --batch_size 16
    done


    for p in 0.2 0.5 0.9 0.95; do
        sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename data/wiki_rankgen/generated/gpt2_xl/v3_sampling_analysis/eags_v3_p_${p}.jsonl --p ${p} --ea_version 4 --entropy_aware_search --fp16
    done
done