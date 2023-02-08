
## Coeffs

### Compute Coeffs Blenderbot-90M:
```bash
    # Degree 1
    python entropy_aware_search/compute_human_entropy_coeffs.py --dataset data/blended_skill_talk/generated/orig.jsonl --max_len 40 --model_name_or_path facebook/blenderbot-90M --is_seq2seq
    
    # human_ma_mean_coeffs: (-0.00102, 2.7048)
    # human_ma_std_coeffs: (0.00735, 0.66831)
    # human_ma_mean_fit_loss: 0.012
    # human_ma_std_fit_loss: 0.002
    
    # Degree 2:
    python entropy_aware_search/compute_human_entropy_coeffs.py --dataset data/blended_skill_talk/generated/orig.jsonl --max_len 40 --model_name_or_path facebook/blenderbot-90M --is_seq2seq --degree 2
    # human_ma_mean_coeffs: (0.00024, -0.01029, 2.76353)
	# human_ma_std_coeffs: (0.00014, 0.00177, 0.70366)
    # human_ma_mean_fit_loss: 0.012
    # human_ma_std_fit_loss: 0.002
```
### Compute Coeffs Blenderbot-1B-distill
```bash
    # Degree 1
    python entropy_aware_search/compute_human_entropy_coeffs.py --dataset data/blended_skill_talk/generated/orig.jsonl --max_len 40 --model_name_or_path facebook/blenderbot-1B-distill --is_seq2seq
    
    # human_ma_mean_coeffs: (-0.01652, 2.57441)
    # human_ma_std_coeffs:  (0.00338, 0.61983)
    # human_ma_mean_fit_loss: 0.022
    # human_ma_std_fit_loss: 0.001
    
    # Degree 2:
    python entropy_aware_search/compute_human_entropy_coeffs.py --dataset data/blended_skill_talk/generated/orig.jsonl --max_len 40 --model_name_or_path facebook/blenderbot-1B-distill --is_seq2seq --degree 2
    # human_ma_mean_coeffs: (0.00064, -0.0414, 2.73196)
	# human_ma_std_coeffs: (-0.00012, 0.0079, 0.59117)
    # human_ma_mean_fit_loss: 0.016
    # human_ma_std_fit_loss: 0.000
```

### Blenderbot-90M
Degree 1: 
Degree 2: 


```bash
sbatch -t 2:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-1B-distill --output_filename data/blended_skill_talk/generated/greedy_bb_1b.jsonl --fp16

sbatch -t 2:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-1B-distill --output_filename data/blended_skill_talk/generated/greedy_bb_1b_3_gram_beam_block.jsonl --no_repeat_ngram_size 3 

sbatch -t 2:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-1B-distill --num_beam 5 --output_filename data/blended_skill_talk/generated/beam_5_bb_1b.jsonl 

sbatch -t 2:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-1B-distill --num_beam 5  --output_filename data/blended_skill_talk/generated/beam_5_bb_1b_3_gram_beam_block.jsonl --no_repeat_ngram_size 3 

sbatch -t 2:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-1B-distill --do_sample --k 30  --output_filename data/blended_skill_talk/generated/topk_30_bb_1b.jsonl

sbatch -t 2:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-1B-distill --do_sample --p 0.9  --output_filename data/blended_skill_talk/generated/top_p_0.9.jsonl 

sbatch -t 2:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-1B-distill --do_sample --p 0.95  --output_filename data/blended_skill_talk/generated/top_p_0.95.jsonl 

sbatch -t 2:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-1B-distill --entropy_aware_search --output_filename data/blended_skill_talk/generated/ead_top_p_0.95.jsonl --p 0.95 --batch_size 1

sbatch -t 2:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-1B-distill --entropy_aware_search --output_filename data/blended_skill_talk/generated/ead_top_k_30.jsonl --k 30 --batch_size 1

sbatch -t 2:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-1B-distill --entropy_aware_search --output_filename data/blended_skill_talk/generated/ead_typical_0.9.jsonl --typical_p 0.9 --batch_size 1
```

```bash
for x in ead_top_p_0.95 ead_top_k_30 ead_typical_0.9; do
    cmd="python dialog/score_generations.py --dataset data/blended_skill_talk/generated/${x}.jsonl --model_name_or_path facebook/blenderbot-1B-distill --is_seq2seq"
    echo $cmd
    sbatch -t 1:00:00 ./launcher_basic.sh $cmd;
done
```
################# Dialog Blenderbot-90M Experiments ####################
```bash
sbatch -t 1:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-90M --output_filename data/blended_skill_talk/generated/greedy_bb_90m.jsonl

sbatch -t 1:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-90M --output_filename data/blended_skill_talk/generated/greedy_bb_90m_3_gram_beam_block.jsonl --no_repeat_ngram_size 3

sbatch -t 1:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-90M --num_beam 5 --output_filename data/blended_skill_talk/generated/beam_5_bb_90m.jsonl

sbatch -t 1:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-90M --num_beam 5  --output_filename data/blended_skill_talk/generated/beam_5_bb_90m_3_gram_beam_block.jsonl --no_repeat_ngram_size 3

sbatch -t 1:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-90M --do_sample --k 30  --output_filename data/blended_skill_talk/generated/topk_30_bb_90m.jsonl

sbatch -t 1:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-90M --do_sample --p 0.9  --output_filename data/blended_skill_talk/generated/top_p_0.9_bb_90m.jsonl

sbatch -t 1:00:00 ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --model_name_or_path facebook/blenderbot-90M --entropy_aware_search --output_filename data/blended_skill_talk/generated/ead_bb_90m.jsonl --fp16 --ea_version 4 --ea_human_mean_coeffs -0.00125 2.7091 --ea_human_std_coeffs 0.00742 0.66802 --p 0.9

```

################### Score Generations #############################
``` bash
for generation in greedy_bb_90m greedy_bb_90m_3_gram_beam_block beam_5_bb_90m beam_5_bb_90m_3_gram_beam_block topk_30_bb_90m top_p_0.9_bb_90m; do 
    echo "Processing ${generation}"
    python dialog/score_generations.py --dataset data/blended_skill_talk/generated/${generation}.jsonl --model_name_or_path facebook/blenderbot-90M --is_seq2seq;
done
```