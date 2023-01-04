sbatch -t 4:00:00 ./launcher_basic.sh python machine_translation/wmt/generate_from_mbart.py --num_beams 5 --output_filename data/wmt17/generated/beam_5_mbart.json

sbatch -t 4:00:00 ./launcher_basic.sh python machine_translation/wmt/generate_from_opus.py --num_beams 5 --output_filename data/wmt17/generated/beam_5_opus.json


python summarization/score_generations.py --dataset data/wmt17/generated/beam_5.json --is_seq2seq --max_source_length 128 --model_name_or_path facebook/mbart-large-50-many-to-one-mmt


python summarization/score_generations.py --dataset data/wmt17/generated/beam_5_mbart.json --is_seq2seq --max_source_length 128 --model_name_or_path facebook/mbart-large-50-many-to-one-mmt

python machine_translation/wmt/score_generations.py --dataset data/wmt17/generated/beam_5_opus.json --is_seq2seq --max_source_length 128 --model_name_or_path Helsinki-NLP/opus-mt-de-en