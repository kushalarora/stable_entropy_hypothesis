

### Compute Human Coeffs
```bash

## For wiki dataset w/ GPT2-XL
python entropy_aware_search/compute_human_entropy_coeffs.py \
    --dataset data/wiki_rankgen//generated/orig.jsonl \
    --max_len 128 --model_name_or_path gpt2-xl

 # human_ma_mean_coeffs: (-0.00277, 2.88702)
 # human_ma_std_coeffs: (-0.00064, 0.91427)
 
 ## Mean and Std. fit loss:
    # human_ma_mean_fit_loss: 0.010
    # human_ma_std_fit_loss: 0.012
```

```bash

```

