#!/bin/bash
set -ex
source ~/envs/seh/bin/activate
model_name=${model_name:="google/gemma-2b"}
batch_size=${batch_size:=32}
max_length=${max_length:=1024}
max_prompt_length=${max_prompt_length:=256}
num_examples=${num_examples:=8192}
generation_key=${generation_key:="generations"}
mkdir -p data/text_completion/corr_analysis/${model_name}/


# Greedy
filename=data/text_completion/corr_analysis/${model_name}/greedy.jsonl
if [[ ! -f "${filename}" ]] || [[ $(($(wc -l < ${filename}) < 5000)) -eq 1 ]];
then
    echo "Running Greedy."
    torchrun --nproc-per-node 8 generate_samples.py --dataset_name wikipedia --model_name_or_path ${model_name} --output_filename ${filename} --bf16 --batch_size ${batch_size} --max_length ${max_length} --max_prompt_length ${max_prompt_length} --num_examples ${num_examples}    
fi


# Beam
for num_beams in 2 4 8 16 32; 
do 
    filename=data/text_completion/corr_analysis/${model_name}/beam_${num_beams}.jsonl
    if [[ ! -f "${filename}" ]] || [[ $(($(wc -l < ${filename}) < 5000)) -eq 1 ]];
    then
        echo "Running Beam Search: ${num_beams}."
        torchrun --nproc-per-node 8 generate_samples.py --dataset_name wikipedia --model_name_or_path ${model_name} --output_filename ${filename} --bf16 --max_length ${max_length} --max_prompt_length ${max_prompt_length} --num_examples ${num_examples} --batch_size $((${batch_size}/${num_beams})) --num_beams ${num_beams}
    fi
done
device=0
for suffix in "greedy" "beam_2" "beam_5" "beam_10" "beam_20":
    filename=data/text_completion/corr_analysis/${model_name}/${suffix}.jsonl
    nohup python text_completion/score_generations.py --model_name_or_path ${model_name} --dataset ${filename} --compute_entropy_voilations --eval_mauve --generation_key ${generation_key} --device ${device} > ${filename}.log 2>&1 &;
    device=$((device+1))
done