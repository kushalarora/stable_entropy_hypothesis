model_lists=(
    # "stabilityai/stablelm-2-1_6b-chat"
    # "stabilityai/stablelm-2-1_6b"
    # "stabilityai/stablelm-2-12b-chat"
    # "stabilityai/stablelm-2-12b"
    # "meta-llama/Meta-Llama-3-8B"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "meta-llama/Meta-Llama-3-70B-Instruct"
    # "meta-llama/Meta-Llama-3-70B"
    # "microsoft/Phi-3-mini-4k-instruct"
    # "google/gemma-2b"
    # "google/gemma-2b-it"
    # "google/gemma-7b"
    # "google/gemma-7b-it"
    # "mistralai/Mistral-7B-v0.1"
    # "mistralai/Mistral-7B-Instruct-v0.2"
    # "mistralai/Mixtral-8x7B-v0.1"
    # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-2-13b-hf"
    "meta-llama/Llama-2-13b-chat-hf"
    "meta-llama/Llama-2-70b-hf"
    "meta-llama/Llama-2-70b-chat-hf"
)

set -eux
for model in "${model_lists[@]}"
do
    output_filename=/data/seh/repeat_vs_models/outputs_${model//"/"/"_"}.jsonl
    echo "Running model: $model | Output file: ${output_filename}"
    torchrun --nproc-per-node 8 text_completion/wiki/generate_from_gpt2.py --num_examples 128 --load_in_8bit --model_name_or_path $model --output_filename $output_filename
    echo "Scoring model: $model | : Output file: ${output_filename}.score"
    python text_completion/score_generations.py --dataset ${output_filename}
done