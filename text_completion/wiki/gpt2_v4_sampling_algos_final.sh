#!/bin/bash
# set -eux
ma_mean=2.71134
ma_std=0.87358
export std_dev=1.0
export typical_p=0.25
export dir=data/wiki_rankgen/generated/gpt2_xl/v4_sampling_analysis_7/
mkdir -p ${dir}
export till=5
export patience_window=5

export run=1
for till in 0 5 10; do
    for std_dev in 0.5 0.8 1.0 1.25; do
        for typical_p in 0.2; do
            export output_filename="${dir}/eags_v4_tau_${typical_p}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}.jsonl"
            # echo "${GENERATION_COMMAND_TEMPLATE} --typical_p ${typical_p}""${SCORING_COMMAND_TEMPLATE}"
            # cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --ea_version 4 --ea_human_mean_coeffs -0.0 -0.00215 2.87399 --ea_human_std_coeffs 2e-05 -0.00373 0.97922 --batch_size 4 --ea_human_entropy_std_band ${std_dev} --ea_patience_window ${patience_window} --ea_only_greedy_till ${till} --output_filename ${output_filename} --seed ${RANDOM} --typical_p ${typical_p}";
            # echo $cmd1;
            # JOBID1=$(sbatch -t 5:00:00 --parsable ./launcher_basic.sh $cmd1)
            # cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
            # sbatch -t 2:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
            # echo $cmd2
            # bash ./text_completion/wiki/gpt2_v4_final_command.sh --typical_p ${typical_p}
            # sbatch -t 6:00:00 ./text_completion/wiki/gpt2_v4_final_command.sh --typical_p ${typical_p}

            export output_filename="${dir}/eags_v4_tau_${typical_p}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}_ea_donot_intervene_for_lower_bound.jsonl"
            # cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --ea_version 4 --ea_human_mean_coeffs -0.0 -0.00215 2.87399 --ea_human_std_coeffs 2e-05 -0.00373 0.97922 --batch_size 4 --ea_human_entropy_std_band ${std_dev} --ea_patience_window ${patience_window} --ea_only_greedy_till ${till} --output_filename ${output_filename} --seed ${RANDOM} --typical_p ${typical_p} --ea_donot_intervene_for_lower_bound";
            # echo $cmd1;
            # JOBID1=$(sbatch -t 5:00:00 --parsable ./launcher_basic.sh $cmd1)
            # cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
            # sbatch -t 2:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
            # echo $cmd2
            sbatch -t 6:00:00 ./text_completion/wiki/gpt2_v4_final_command.sh --typical_p ${typical_p} --ea_donot_intervene_for_lower_bound
        done

        for k in 30; do
            export output_filename="${dir}/eags_v4_k_${k}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}.jsonl"
        #     cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --ea_version 4 --ea_human_mean_coeffs -0.0 -0.00215 2.87399 --ea_human_std_coeffs 2e-05 -0.00373 0.97922 --batch_size 4 --ea_human_entropy_std_band ${std_dev} --ea_patience_window ${patience_window} --ea_only_greedy_till ${till} --output_filename ${output_filename} --seed ${RANDOM} --k ${k} ";
        #     echo $cmd1;
        #     JOBID1=$(sbatch -p ${partition} --mem 24GB -t 6:00:00 --parsable ./launcher_basic.sh $cmd1)
        #     cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
        #     sbatch -p ${partition} -t 2:00:00 --mem 24GB  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
        #     echo $cmd2
            # sbatch -t 6:00:00 ./text_completion/wiki/gpt2_v4_final_command.sh --k ${k}

            export output_filename="${dir}/eags_v4_k_${k}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}_ea_donot_intervene_for_lower_bound.jsonl"
        #     cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --ea_version 4 --ea_human_mean_coeffs -0.0 -0.00215 2.87399 --ea_human_std_coeffs 2e-05 -0.00373 0.97922 --batch_size 4 --ea_human_entropy_std_band ${std_dev} --ea_patience_window ${patience_window} --ea_only_greedy_till ${till} --output_filename ${output_filename} --seed ${RANDOM} --k ${k} --ea_donot_intervene_for_lower_bound";
        #     echo $cmd1;
        #     JOBID1=$(sbatch -t 5:00:00 --parsable ./launcher_basic.sh $cmd1)
        #     cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
        #     sbatch -t 2:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
        #     echo $cmd2
            sbatch -t 6:00:00 ./text_completion/wiki/gpt2_v4_final_command.sh --k ${k} --ea_donot_intervene_for_lower_bound
        done

        export output_filename="${dir}/eags_v4_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}_ea_donot_intervene_for_upper_bound.jsonl"
        # cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --ea_version 4 --ea_human_mean_coeffs -0.0 -0.00215 2.87399 --ea_human_std_coeffs 2e-05 -0.00373 0.97922 --batch_size 4 --ea_human_entropy_std_band ${std_dev} --ea_patience_window ${patience_window} --ea_only_greedy_till ${till} --output_filename ${output_filename} --seed ${RANDOM} --ea_donot_intervene_for_upper_bound";
        # echo $cmd1;
        # JOBID1=$(sbatch -t 20:00:00 --parsable ./launcher_basic.sh $cmd1)
        # cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
        # sbatch -t 2:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
        # echo $cmd2
        # sbatch -t 20:00:00 ./text_completion/wiki/gpt2_v4_final_command.sh --ea_donot_intervene_for_upper_bound
    done
done