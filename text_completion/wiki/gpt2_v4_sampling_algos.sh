!#/bin/bash

### DEPRECATED #####
for std_dev in 0.8 1.0 1.2 1.5; do
    for patience_window in 3 4 5 6 8; do
        for tau in 0.2 0.5 0.9 0.95; do
            cmd="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --typical_p ${tau} --version 4 --entropy_aware_search --batch_size 4 --ea_upper_limit_coeffs ${upper_entropy_coeffs} --ea_lower_limit_coeffs ${lower_entropy_coeffs} --ea_patience_window ${patience_window}  --output_filename data/wiki_rankgen/generated/gpt2_xl/v4_sampling_analysis/eags_v4_tau_${tau}_std_dev_${std_dev}_patience_${patience_window}.jsonl";
                echo $cmd;
            sbatch -t 10:00:00 ./launcher_basic.sh $cmd
        done
    done
dones