#!/bin/bash
set -ex

mkdir -p data/wiki_rankgen/corr_analysis/gpt2_xl/

# Greedy
filename=data/wiki_rankgen/corr_analysis/gpt2_xl/greedy.jsonl
if [[ ! -f "${filename}" ]] || [[ $(($(wc -l < ${filename}) < 5000)) -eq 1 ]];
then
    echo "Running Greedy."
    JOBID1=$(sbatch -t 4:00:00  --parsable  ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename ${filename} --fp16); 
    echo $JOBID1
    sbatch -t 2:00:00 --parsable --dependency=afterok:${JOBID1} launcher_basic.sh python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${filename};
fi

# Beam
for num_beams in 2 5 10 20; 
do 
    filename=data/wiki_rankgen/corr_analysis/gpt2_xl/beam_${num_beams}.jsonl
    if [[ ! -f "${filename}" ]] || [[ $(($(wc -l < ${filename}) < 5000)) -eq 1 ]];
    then
        echo "Running Beam Search: ${num_beams}."
        JOBID1=$(sbatch -t 48:00:00 --parsable ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename ${filename} --num_beams ${num_beams} --batch_size $((64/${num_beams})) --fp16);
        echo $JOBID1
        sbatch -t 2:00:00 --parsable --dependency=afterok:${JOBID1} launcher_basic.sh python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${filename};
    fi
done

for run in 1 2 3 4 5; do
    for p in 0.25 0.5 0.75 0.9 0.95; 
    # for p in 0.15 0.4; 
    do
        filename="data/wiki_rankgen/corr_analysis/gpt2_xl/top_p_${p}_run_${run}.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running Nucleus Sampling: run=${run} || top-p=${p}."
            JOBID1=$(sbatch -t 4:00:00 --parsable ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename ${filename}  --p ${p} --do_sample --fp16 --seed ${seed});
            echo $JOBID1
            sbatch -t 2:00:00 --parsable --dependency=afterok:${JOBID1} launcher_basic.sh python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${filename}
        fi
    done
# done
    for k in 5 10 30 50 100; 
    do
        filename="data/wiki_rankgen/corr_analysis/gpt2_xl/top_k_${k}_run_${run}.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running Top-k Sampling: run=${run} || top-k=${k}."
            JOBID1=$(sbatch -t 4:00:00 --parsable ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename ${filename}  --k ${k} --do_sample --fp16 --seed ${seed});
            echo $JOBID1
            sbatch -t 2:00:00 --parsable --dependency=afterok:${JOBID1} launcher_basic.sh python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${filename}
        fi
    done

    for tau in 0.2 0.25 0.5 0.75 0.9 0.95; 
    do
        filename="data/wiki_rankgen/corr_analysis/gpt2_xl/typical_p_${tau}_run_${run}.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running Typical Sampling: run=${run} || tau=${tau}."
            JOBID1=$(sbatch -t 8:00:00 --parsable ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename ${filename}  --typical_p ${tau} --do_sample --seed ${seed} --batch_size 16);
            echo $JOBID1
            sbatch -t 2:00:00 --parsable --dependency=afterok:${JOBID1} launcher_basic.sh python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${filename};
        fi
    done

    for temp in 0.001 0.01 0.1 0.2 0.5 0.8 1.0 1.2 1.5 3.0 1.0;
    do
        filename="data/wiki_rankgen/corr_analysis/gpt2_xl/temp_${temp}_run_${run}.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running Temperature Sampling: run=${run} || temp=${temp}."
            JOBID1=$(sbatch -t 4:00:00 --parsable ./launcher_basic.sh python text_completion/wiki/generate_from_gpt2.py --model_name_or_path gpt2-xl --output_filename ${filename} --temperature ${temp} --do_sample --seed ${seed} --fp16);
            echo $JOBID1
            sbatch -t 2:00:00 --parsable --dependency=afterok:${JOBID1} launcher_basic.sh python text_completion/score_generations.py --model_name_or_path gpt2-xl --dataset ${filename};
        fi
    done
done