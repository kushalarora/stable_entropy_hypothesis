!#/bin/bash
ma_mean=2.71134
ma_std=0.87358
for upper_entropy_coeffs in 3.59 4.02 4.46 5.33 5.77; do
    for lower_entropy_coeffs in 1.837 1.401 0.9641; do
        for only_greedy_till in 0 5 10; do
            cmd="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --typical_p 0.2 --version 4 --entropy_aware_search --batch_size 4 --ea_upper_limit_coeffs ${upper_entropy_coeffs} --ea_lower_limit_coeffs ${lower_entropy_coeffs} --only_greedy_till ${only_greedy_till}  --output_filename data/wiki_rankgen/generated/gpt2_xl/v4_1_sampling_analysis/eags_v4_typical_tau_0.2_upper_${upper_entropy_coeffs}_lower_${lower_entropy_coeffs}_only_greedy_till_${only_greedy_till}.jsonl";
            echo $cmd;
            sbatch -t 24:00:00 ./launcher_basic.sh $cmd
        done
    done
done