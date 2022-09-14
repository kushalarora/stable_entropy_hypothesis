# Entropy Aware Search

## Installation

Please use python>=3.8

python -m venv ${HOME}/envs/ews
source ${HOME}/envs/ews/bin/activate

pip install -e .

## Writing Prompts Experiments


### Prepare Dataset

Download dataset from https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz. 
```bash
mkdir data
cd data
wget https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz | tar -xvf 

cd -;

python text_completion/writing_prompts/prepare_dataset.py data/WritingPrompts/
```
### Fine-tune GPT-2 on writing prompt dataset.
```!#/bin/python
python text_completion/writing_prompts/fine_tune_writing_prompts.py
```

### Generate from Fine-tuned GPT2 model
```bash
sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-18-2022-05-33-01/checkpoint-55000/ --output_filename data/writingPrompts/generated/finetuned/v2/greedy.csv

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-18-2022-05-33-01/checkpoint-55000/ --output_filename data/writingPrompts/generated/finetuned/v2/top_p_0.9.csv  --p 0.9 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-18-2022-05-33-01/checkpoint-55000/ --output_filename data/writingPrompts/generated/finetuned/v2/top_p_0.95.csv  --p 0.95 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-18-2022-05-33-01/checkpoint-55000/ --output_filename data/writingPrompts/generated/finetuned/v2/top_k_30.csv  --k 30 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-18-2022-05-33-01/checkpoint-55000/ --output_filename data/writingPrompts/generated/finetuned/v2/top_k_40.csv  --k 40 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-18-2022-05-33-01/checkpoint-55000/ --output_filename data/writingPrompts/generated/finetuned/v2/typical_p_0.95.csv  --typical_p 0.95 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-18-2022-05-33-01/checkpoint-55000/ --output_filename data/writingPrompts/generated/finetuned/v2/typical_p_0.2.csv  --typical_p 0.2 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-18-2022-05-33-01/checkpoint-55000/ --output_filename data/writingPrompts/generated/finetuned/v2/temp_1.csv  --temperature 1 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-18-2022-05-33-01/checkpoint-55000/ --output_filename data/writingPrompts/generated/finetuned/v2/temp_1_2.csv  --temperature 1.2 --do_sample

sbatch -t 24:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-18-2022-05-33-01/checkpoint-55000/ --output_filename data/writingPrompts/generated/finetuned/v2/beam_10.csv --num_beams 10 --batch_size 8


```


### Generate from Original GPT2 model
```bash
sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path gpt2 --output_filename data/writingPrompts/generated/gpt2/greedy.csv

sbatch -t 48:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path gpt2 --output_filename data/writingPrompts/generated/gpt2/beam_10.csv --num_beams 10 --batch_size 8

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path gpt2 --output_filename data/writingPrompts/generated/gpt2/top_p_0.9.csv  --p 0.9 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path gpt2 --output_filename data/writingPrompts/generated/gpt2/top_p_0.95.csv  --p 0.95 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path gpt2 --output_filename data/writingPrompts/generated/gpt2/top_k_30.csv  --k 30 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path gpt2 --output_filename data/writingPrompts/generated/gpt2/top_k_40.csv  --k 40 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path gpt2 --output_filename data/writingPrompts/generated/gpt2/typical_p_0.95.csv  --typical_p 0.95 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path gpt2 --output_filename data/writingPrompts/generated/gpt2/typical_p_0.2.csv  --typical_p 0.2 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path gpt2 --output_filename data/writingPrompts/generated/gpt2/temp_1.csv  --temperature 1 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path gpt2 --output_filename data/writingPrompts/generated/gpt2/temp_1_2.csv  --temperature 1.2 --do_sample
```

## BigBird Pegasus ArXiv Summarization Experiments
### Generate from Fine-tuned GPT2 model
```bash
sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-14-2022-11-03/checkpoint-71000/ --output_filename data/writingPrompts/generated/finetuned/greedy.csv

sbatch -t 24:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-14-2022-11-03/checkpoint-71000/ --output_filename data/writingPrompts/generated/finetuned/beam_10.csv --num_beams 10 --batch_size 8

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-14-2022-11-03/checkpoint-71000/ --output_filename data/writingPrompts/generated/finetuned/top_p_0.9.csv  --p 0.9 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-14-2022-11-03/checkpoint-71000/ --output_filename data/writingPrompts/generated/finetuned/top_p_0.95.csv  --p 0.95 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-14-2022-11-03/checkpoint-71000/ --output_filename data/writingPrompts/generated/finetuned/top_k_30.csv  --k 30 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-14-2022-11-03/checkpoint-71000/ --output_filename data/writingPrompts/generated/finetuned/top_k_40.csv  --k 40 --do_sample

sbatch -t 4:00:00 ./launcher_basic.sh python text_completion/writing_prompts/generate_from_gpt2.py --model_name_or_path /home/mila/a/arorakus/scratch/ews/finetuned_writing_prompts/08-14-2022-11-03/checkpoint-71000/ --output_filename data/writingPrompts/generated/finetuned/typical_p_0.95.csv  --typical_p 0.95 --do_sample
```


### Generate from CNN-DailyMail Pegasus
```bash
sbatch -t 1-00:00:00 ./launcher_basic.sh python summarization/cnn_dm/generate_from_cnn_dm.py --output_filename data/cnn_dm_pegasus/generated/eabs_beams_5_width_5_alpha_5.csv --entropy_aware_search --num_beams 5

sbatch -t 1-00:00:00 ./launcher_basic.sh python summarization/cnn_dm/generate_from_cnn_dm.py --output_filename data/cnn_dm_pegasus/generated/beams_5.csv --num_beams 5
```
