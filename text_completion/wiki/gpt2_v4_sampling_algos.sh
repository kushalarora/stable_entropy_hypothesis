!#/bin/bash

for upper_entropy_coeffs in 3.5 4 4.5 6; do
    for lower_entropy_coeffs in 1 1.5 2 2.5; do
        for patience_window in 3 4 5 6 8; do
            for tau in 0.2 0.5; do
                cmd="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --typical_p ${tau} --version 4 --entropy_aware_search --batch_size 4 --ea_upper_limit_coeffs ${upper_entropy_coeffs} --ea_lower_limit_coeffs ${lower_entropy_coeffs} --patience_window ${patience_window}  --output_filename data/wiki_rankgen/generated/gpt2_xl/v4_sampling_analysis/eags_v4_typical_tau_${tau}_upper_${upper_entropy_coeffs}_lower_${lower_entropy_coeffs}_patience_${patience_window}.jsonl";
                echo $cmd;
                sbatch -t 10:00:00 ./launcher_basic.sh $cmd
            done
        done
    done
done