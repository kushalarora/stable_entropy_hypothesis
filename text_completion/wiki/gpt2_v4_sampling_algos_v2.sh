!#/bin/bash
ma_mean=2.71134
ma_std=0.87358
std_dev=1.0
for run in 1 2 3 4 5; do
    for till in 0 5 10; do
        for p in k; do
            output_filename="data/wiki_rankgen/generated/gpt2_xl/v4_sampling_analysis_2/eags_v4_p_${p}_std_dev_${std_dev}_till_${till}_run_${run}.jsonl"
            cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --p ${p} --ea_version 4 --entropy_aware_search --batch_size 8 --ea_human_entropy_std_band ${std_dev} --ea_only_greedy_till ${till}  --output_filename ${output_filename} --fp16";
            echo $cmd1;
            JOBID1=$(sbatch -t 8:00:00 --parsable ./launcher_basic.sh $cmd1)
            cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
            sbatch -t 2:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
        done
    done
done