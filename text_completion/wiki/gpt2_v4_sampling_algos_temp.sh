!#/bin/bash
ma_mean=2.71134
ma_std=0.87358
for std_dev in 0.8 1.0 1.2 1.5; do
    for temp in 0.01 0.1 0.2 0.5 0.9; do
# for std_dev in 0.8; do 
    # for temp in 0.1; do
        output_filename="data/wiki_rankgen/generated/gpt2_xl/v4_sampling_analysis_2/eags_v4_typical_temp_${temp}_std_dev_${std_dev}.jsonl"
        cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --temperature ${temp} --ea_version 4 --entropy_aware_search --batch_size 1 --ea_human_entropy_std_band ${std_dev} --ea_only_greedy_till 0  --output_filename ${output_filename}";
        echo $cmd1;
        JOBID1=$(sbatch -t 24:00:00 --parsable ./launcher_basic.sh $cmd1  )
        echo $JOBID1;
        cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
        sbatch -t 2:00:00 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh  $cmd2
        echo $cmd2;
    done
done