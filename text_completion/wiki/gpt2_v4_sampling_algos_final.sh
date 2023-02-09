#!/bin/bash
# set -eux
export typical_p=0.2
export k=30
export dir=data/wiki_rankgen/generated/gpt2_xl/sampling_analysis/
mkdir -p ${dir}
export patience_window=5

for till in 0 5 10; do
    for std_dev in 0.5 0.8 1.0 1.25; do
        for run in 1 2 3 4 5; do
            export output_filename="${dir}/eags_v4_tau_${typical_p}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}.jsonl"
            sbatch -t 6:00:00 ./text_completion/wiki/gpt2_v4_final_command.sh --typical_p ${typical_p}

            export output_filename="${dir}/eags_v4_tau_${typical_p}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}_ea_donot_intervene_for_lower_bound.jsonl"
            sbatch -t 6:00:00 ./text_completion/wiki/gpt2_v4_final_command.sh --typical_p ${typical_p} --ea_donot_intervene_for_lower_bound
        
            export output_filename="${dir}/eags_v4_k_${k}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}.jsonl"
            sbatch -t 6:00:00 ./text_completion/wiki/gpt2_v4_final_command.sh --k ${k}

            export output_filename="${dir}/eags_v4_k_${k}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}_ea_donot_intervene_for_lower_bound.jsonl"
            sbatch -t 6:00:00 ./text_completion/wiki/gpt2_v4_final_command.sh --k ${k} --ea_donot_intervene_for_lower_bound
        done

        export output_filename="${dir}/eags_v4_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}_ea_donot_intervene_for_upper_bound.jsonl"
        sbatch -t 20:00:00 ./text_completion/wiki/gpt2_v4_final_command.sh --ea_donot_intervene_for_upper_bound
    done
done