!#/bin/bash
# set -eux
ma_mean=2.71134
ma_std=0.87358
std_dev=0.8
typical_p=0.25
mkdir -p data/wiki_rankgen/generated/gpt2_xl/v4_sampling_analysis_3/
for run in 1 2 3; do
    for till in 5; do
        for patience_window in 3 4 5; do
            seed=$((1 + RANDOM % 1000))
            output_filename="data/wiki_rankgen/generated/gpt2_xl/v4_sampling_analysis_3/eags_v4_tau_${typical_p}_std_dev_${std_dev}_till_${till}_patience_${patience_window}_run_${run}.jsonl"
            cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --typical_p ${typical_p} --ea_version 4 --batch_size 8 --ea_patience_window ${patience_window} --ea_only_greedy_till ${till}  --output_filename ${output_filename} --fp16 --seed ${RANDOM} --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427 --ea_human_entropy_std_band ${std_dev}";
            echo $cmd1;
            JOBID1=$(sbatch -t 4:00:00 --parsable ./launcher_basic.sh $cmd1)
            cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
            sbatch -t 2:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
            echo $cmd2
        done
    done
done