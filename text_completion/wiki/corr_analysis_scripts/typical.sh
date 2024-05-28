#!/bin/bash
set -ex
source ~/envs/seh/bin/activate
model_name=${model_name:="google/gemma-2b"}
batch_size=${batch_size:=32}
max_length=${max_length:=1024}
max_prompt_length=${max_prompt_length:=32}
num_examples=${num_examples:=8192}
generation_key=${generation_key:="generations"}
mkdir -p data/text_completion/corr_analysis/${model_name}/

for run in 1 2 3 4 5; do
    for tau in 0.15 0.25 0.4 0.5 0.75 0.85 0.9 0.95; do
        filename="data/text_completion/corr_analysis/${model_name}/typical_p_${tau}_run_${run}.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running Nucleus Sampling: run=${run} || top-p=${p}."
            torchrun --nproc-per-node 8 generate_samples.py --dataset_name wikipedia --model_name_or_path ${model_name} --output_filename ${filename} --bf16 --batch_size ${batch_size} --max_length ${max_length} --max_prompt_length ${max_prompt_length} --num_examples ${num_examples} --typical_p ${tau} --do_sample --seed ${seed}
        fi
    done

    device=0
    for tau in 0.15 0.25 0.4 0.5 0.75 0.85 0.9 0.95; do
        filename="data/text_completion/corr_analysis/${model_name}/typical_p_${tau}_run_${run}.jsonl"
        nohup python text_completion/score_generations.py --model_name_or_path ${model_name} --dataset ${filename} --compute_entropy_voilations --eval_mauve --generation_key ${generation_key} --device ${device} > ${filename}.log 2>&1 &
        device=$((device+1))
    done
    sleep 1800
done