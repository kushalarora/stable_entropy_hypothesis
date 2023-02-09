#!/bin/bash
set -ex

directory="data/blended_skill_talk/generated/BB_90M_degree_2/"
mkdir -p ${directory}
export patience_window=5
export model_name="facebook/blenderbot-90M"
run=1
for till in 0 5 10; do
    for std_dev in 0.25 0.5 0.8 1; do
        export till=${till}
        export std_dev=${std_dev}
        export output_filename="${directory}/ead_std_dev_${std_dev}_p_0.9_till_${till}_patience_${patience_window}_run_${run}_deg_2.jsonl"
        sbatch -p main -t 1:00:00 ./dialog/blended_skill_talk/bb_90M_final_command.sh --p 0.9

        export output_filename="${directory}/ead_std_dev_${std_dev}_p_0.9_till_${till}_patience_${patience_window}_run_${run}_deg_2_ea_donot_intervene_for_lower_bound.jsonl"
        sbatch  -p main -t 1:00:00 ./dialog/blended_skill_talk/bb_90M_final_command.sh --p 0.9 --ea_donot_intervene_for_lower_bound

        export output_filename="${directory}/ead_std_dev_${std_dev}_k_30_till_${till}_patience_${patience_window}_run_${run}_deg_2.jsonl"
        sbatch  -p main -t 1:00:00 ./dialog/blended_skill_talk/bb_90M_final_command.sh --k 30

        export output_filename="${directory}/ead_std_dev_${std_dev}_k_30_till_${till}_patience_${patience_window}_run_${run}_deg_2_ea_donot_intervene_for_lower_bound.jsonl"
        sbatch  -p main -t 1:00:00 ./dialog/blended_skill_talk/bb_90M_final_command.sh --k 30 --ea_donot_intervene_for_lower_bound

        export output_filename="${directory}/ead_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}ea_donot_intervene_for_upper_bound.jsonl"
        sbatch  -p main -t 5:00:00 ./dialog/blended_skill_talk/bb_90M_final_command.sh --ea_donot_intervene_for_upper_bound
    done
done