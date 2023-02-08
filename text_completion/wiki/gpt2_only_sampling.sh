!#/bin/bash
ma_mean=2.71134
ma_std=0.87358
dir=data/wiki_rankgen/generated/gpt2_xl/eas/
mkdir -p ${dir}
for run in 1 2 3; do
    for std_dev in 0 0.25 0.5 0.75 1 1.5; do
        output_filename="${dir}/eas_std_dev_${std_dev}_run_${run}.jsonl"
        cmd1="python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --do_sample --batch_size 64  --ea_human_mean_coeffs -0.00277 2.88702 --ea_human_std_coeffs -0.00064 0.91427  --ea_human_entropy_std_band ${std_dev} --output_filename ${output_filename} --fp16 --seed ${RANDOM}";
        echo $cmd1;
        JOBID1=$(sbatch -t 1:00:00 --parsable ./launcher_basic.sh $cmd1)
        cmd2="python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${output_filename}"
        echo $cmd2
        sbatch -t 1:00:00  --dependency=afterok:${JOBID1} ./launcher_basic.sh $cmd2
    done
done