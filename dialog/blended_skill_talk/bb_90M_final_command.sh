#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --nodes=1
#SBATCH --mem=24000M
#SBATCH --mail-type=ALL,TIME_LIMIT,BEGIN,END,FAIL
#SBATCH --mail-user=arorakus@mila.quebec
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:3g.39gb:1
#SBATCH -o logs/slurm-%x-%j.out
#SBATCH -e logs/slurm-%x-%j.err
###########################

set -x
module load cuda
module load python/3.8 
module load libffi

export NUM_GPUS=${NUM_GPUS:=2}
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.:${PYTHONPATH}
source ${HOME}/scratch/envs/ews2/bin/activate

python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path ${model_name} --batch_size 64 --ea_human_mean_coeffs 0.00024 -0.01029 2.76353 --ea_human_std_coeffs 0.00014 0.00177 0.70366 --ea_human_entropy_std_band ${std_dev} --ea_patience_window ${patience_window} --ea_only_greedy_till ${till} --output_filename ${output_filename} --seed ${RANDOM} $@ --min_length 20

python dialog/score_generations.py  --is_seq2seq --model_name_or_path ${model_name} --dataset ${output_filename}