!#/bin/bash
# set -eux
ma_mean=2.71134
ma_std=0.87358
std_dev=1.0
typical_p=0.25
dir=data/wiki_rankgen/generated/gpt2_xl/v4_sampling_analysis_6/
mkdir -p ${dir}
till=5
patience_window=5
partition=long
for run in 1; do
    for std_dev in 0.8; do
        # output_filename="${dir}/eags_v4_eas_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}.jsonl"
        # cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --ea_version 4 --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --ea_human_entropy_std_band ${std_dev} --entropy_aware_sampling --batch_size 4 --ea_patience_window ${patience_window} --ea_only_greedy_till ${till}  --output_filename ${output_filename} --seed ${RANDOM}";
        # echo $cmd1;
        # JOBID1=$(sbatch -t 5:00:00 --parsable ./launcher_basic.sh $cmd1)
        # cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
        # sbatch -t 2:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
        # echo $cmd2
        
        # output_filename="${dir}/eags_v4_eas_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}_ea_donot_intervene_for_lower_bound.jsonl"
        # cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --ea_version 4 --entropy_aware_sampling --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --ea_human_entropy_std_band ${std_dev} --batch_size 4 --ea_patience_window ${patience_window} --ea_only_greedy_till ${till}  --output_filename ${output_filename} --seed ${RANDOM} --ea_donot_intervene_for_lower_bound";
        # echo $cmd1;
        # JOBID1=$(sbatch -t 5:00:00 --parsable ./launcher_basic.sh $cmd1)
        # cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
        # sbatch -t 2:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
        # echo $cmd2

        # output_filename="${dir}/eags_v4_eas_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}_ea_donot_intervene_for_upper_bound.jsonl"
        # cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --ea_version 4 --entropy_aware_sampling --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --ea_human_entropy_std_band ${std_dev} --batch_size 4 --ea_patience_window ${patience_window} --ea_only_greedy_till ${till}  --output_filename ${output_filename} --seed ${RANDOM} --ea_donot_intervene_for_upper_bound";
        # echo $cmd1;
        # JOBID1=$(sbatch -p ${partition} --mem 24GB -t 24:00:00 --parsable ./launcher_basic.sh $cmd1)
        # cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
        # sbatch -p ${partition} -t 2:00:00 --mem 24GB  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
        # echo $cmd2

        for typical_p in 0.2; do
            output_filename="${dir}/eags_v4_tau_${typical_p}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}.jsonl"
            cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --typical_p ${typical_p} --ea_version 4 --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --ea_human_entropy_std_band ${std_dev} --batch_size 4 --ea_patience_window ${patience_window} --ea_only_greedy_till ${till}  --output_filename ${output_filename} --seed ${RANDOM}";
            echo $cmd1;
            JOBID1=$(sbatch -p ${partition} --mem 24GB -t 6:00:00 --parsable ./launcher_basic.sh $cmd1)
            cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
            sbatch -p ${partition} -t 2:00:00 --mem 24GB  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
            echo $cmd2

            output_filename="${dir}/eags_v4_tau_${typical_p}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}_ea_donot_intervene_for_lower_bound.jsonl"
            cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --typical_p ${typical_p} --ea_version 4 --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --ea_human_entropy_std_band ${std_dev} --batch_size 4 --ea_patience_window ${patience_window} --ea_only_greedy_till ${till}  --output_filename ${output_filename} --seed ${RANDOM} --ea_donot_intervene_for_lower_bound";
            echo $cmd1;
            JOBID1=$(sbatch -t 5:00:00 --parsable ./launcher_basic.sh $cmd1)
            cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
            sbatch -t 2:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
            echo $cmd2
        done

        for k in 30; do
            output_filename="${dir}/eags_v4_tau_${typical_p}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}.jsonl"
            cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --k ${k} --ea_version 4 --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --ea_human_entropy_std_band ${std_dev} --batch_size 4 --ea_patience_window ${patience_window} --ea_only_greedy_till ${till}  --output_filename ${output_filename} --seed ${RANDOM}";
            echo $cmd1;
            JOBID1=$(sbatch -p ${partition} --mem 24GB -t 6:00:00 --parsable ./launcher_basic.sh $cmd1)
            cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
            sbatch -p ${partition} -t 2:00:00 --mem 24GB  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
            echo $cmd2

            output_filename="${dir}/eags_v4_tau_${typical_p}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}_ea_donot_intervene_for_lower_bound.jsonl"
            cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --k ${k} --ea_version 4 --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --ea_human_entropy_std_band ${std_dev} --batch_size 4 --ea_patience_window ${patience_window} --ea_only_greedy_till ${till}  --output_filename ${output_filename} --seed ${RANDOM} --ea_donot_intervene_for_lower_bound";
            echo $cmd1;
            JOBID1=$(sbatch -t 5:00:00 --parsable ./launcher_basic.sh $cmd1)
            cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
            sbatch -t 2:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
            echo $cmd2
        done

    done
done