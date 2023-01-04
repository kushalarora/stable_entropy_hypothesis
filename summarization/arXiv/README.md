sbatch -t 6:00:00 ./launcher_basic.sh python summarization/arXiv/generate_from_pegasus.py --num_beams 5 --output_filename data/arxiv_pegasus/generated/beam_5.json


python summarization/score_generations.py --dataset data/arxiv_pegasus/generated/beam_5.json --model_name_or_path google/bigbird-pegasus-large-arxiv --max_source_len 4096 --max_len 1024