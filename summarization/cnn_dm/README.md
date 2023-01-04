sbatch -t 4:00:00 ./launcher_basic.sh python summarization/cnn_dm/generate_from_pegasus.py --num_beams 5 --output_filename data/cnn_dm_pegasus/generated/beam_5.json
sbatch -t 4:00:00 ./launcher_basic.sh python summarization/cnn_dm/generate_from_bart.py --num_beams 5 --output_filename data/cnn_dm_bart/generated/beam_5.json


python summarization/score_generations.py --dataset data/cnn_dm_bart/generated/beam_5.json --model_name_or_path facebook/bart-large-cnn --is_seq2seq --max_source_len 256

python summarization/score_generations.py --dataset data/cnn_dm_pegasus/generated/beam_5.json --model_name_or_path google/pegasus-cnn_dailymail --is_seq2seq --max_source_len 256

