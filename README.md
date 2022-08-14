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

### Generate dataset
```
```



