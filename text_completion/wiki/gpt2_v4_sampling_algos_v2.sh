!#/bin/bash
ma_mean=2.71134
ma_std=0.87358
for std_dev in 0.8 1.0 1.2 1.5; do
    for tau in 0.2 0.5 0.9 0.95; do
        output_filename="data/wiki_rankgen/generated/gpt2_xl/v4_sampling_analysis_2/eags_v4_typical_tau_${tau}_std_dev_${std_dev}.jsonl"
        cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --typical_p ${tau} --ea_version 4 --entropy_aware_search --batch_size 1 --ea_human_entropy_std_band ${std_dev} --ea_only_greedy_till 0  --output_filename ${output_filename}";
        echo $cmd;
        JOBID1=$(sbatch -t 24:00:00 ./launcher_basic.sh $cmd1  --parsable)
        cmd2="python text_completion/score_generations.py --dataset ${x} --model_name_or_path gpt2-xl --dependency=afterok:${JOBID1}"
        sbatch -t 2:00:00 ./launcher_basic.sh $cmd2
    done
done