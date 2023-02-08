#!/bin/bash
set -ex

directory="data/blended_skill_talk/generated/BB_1B_degree_2/"
mkdir -p ${directory}
export till=5
export patience_window=5
export model_name="facebook/blenderbot-1B-distill"

for run in 1; do
    for till in 0 5 10; do
    for std_dev in 0.25 0.5 0.8 1; do
        export output_filename="${directory}/ead_std_dev_${std_dev}_k_30_till_${till}_patience_${patience_window}_run_${run}_degree_2.jsonl"
        sbatch -t 6:00:00 ./dialog/blended_skill_talk/bb_final_command.sh --k 30

        output_filename="${directory}/ead_std_dev_${std_dev}_k_30_run_${run}_degree_1_ea_donot_intervene_for_lower_bound.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${output_filename}" ] || [ $(($(wc -l < "${output_filename}") < 5000)) -eq 1 ];
        then
            echo "Running EAD: run=${run} || std_dev=${std_dev}."
            JOBID1=$(sbatch -t 45:00  --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path   --output_output_filename ${output_filename}  --seed ${seed} --batch_size 64 --ea_human_mean_coeffs -0.00102 2.7048 --ea_human_std_coeffs 0.00735 0.66831 --ea_human_entropy_std_band ${std_dev} --k 30  --ea_donot_intervene_for_lower_bound);
            echo $JOBID1
            sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${output_filename} --is_seq2seq;
        fi

        output_filename="${directory}/ead_std_dev_${std_dev}_p_0.9_run_${run}_degree_1.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${output_filename}" ] || [ $(($(wc -l < "${output_filename}") < 5000)) -eq 1 ];
        then
            echo "Running EAD: run=${run} || std_dev=${std_dev}."
            JOBID1=$(sbatch -t 45:00  --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_output_filename ${output_filename}  --seed ${seed} --batch_size 64 --ea_human_mean_coeffs -0.00102 2.7048 --ea_human_std_coeffs 0.00735 0.66831 --ea_human_entropy_std_band ${std_dev} --p 0.9);
            echo $JOBID1
            sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${output_filename} --is_seq2seq;
        fi

        output_filename="${directory}/ead_std_dev_${std_dev}_p_0.9_run_${run}_degree_1_ea_donot_intervene_for_lower_bound.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${output_filename}" ] || [ $(($(wc -l < "${output_filename}") < 5000)) -eq 1 ];
        then
            echo "Running EAD: run=${run} || std_dev=${std_dev}."
            JOBID1=$(sbatch -t 45:00  --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_output_filename ${output_filename}  --seed ${seed} --batch_size 64 --ea_human_mean_coeffs -0.00102 2.7048 --ea_human_std_coeffs 0.00735 0.66831 --ea_human_entropy_std_band ${std_dev} --p 0.9  --ea_donot_intervene_for_lower_bound);
            echo $JOBID1
            sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${output_filename} --is_seq2seq;
        fi
    done
done

for std_dev in 0.25 0.5 0.75 1; 
do
    output_filename="${directory}/ead_std_dev_${std_dev}_k_30_run_${run}_degree_1_ea_donot_intervene_for_upper_bound.jsonl"
    echo "Running EAD: run=${run} || std_dev=${std_dev}."
    JOBID1=$(sbatch -t 45:00 --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_output_filename ${output_filename}  --seed ${seed} --batch_size 64 --ea_human_mean_coeffs -0.00102 2.7048 --ea_human_std_coeffs 0.00735 0.66831 --ea_human_entropy_std_band ${std_dev} --ea_donot_intervene_for_upper_bound);
    echo $JOBID1
    sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py  --is_seq2seq --model_name_or_path facebook/blenderbot-1B-distill --dataset ${output_filename};
done