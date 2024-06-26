# The Stable Entropy Hypothesis and Entropy-Aware Decoding
This repository is the official implementation of [The Stable Entropy Hypothesis and Entropy Aware Decoding](https://arxiv.org/abs/<tobefilled>). 

<p align="center">
<img src="images/stable_entropy_hypothesis_annotated_v2.png" align="middle" width=50% height=50%>
</p>

This paper introduces the concept of the stable entropy baseline and the stable entropy zone, and then use these definitions to posit the stable entropy hypothesis. Finally, we leverage the stable entropy hypothesis to propose an entropy-aware decoding method that generates fluent and non degenerate text in open-ended generation settings while acting greedily most of the time.

### The Stable Entropy Zone:
We define the **stable entropy baseline** as the smoothed mean entropy of the conditional distribution of the model under target context distribution, i.e., under target data. We posit that the stable entropy baseline is flat across the length of the generation. 

Next, we define an area around the** stable entropy baseline** spanned by its standard deviation as the **stable entropy zone**. We hypothesize that the stable entropy zone is narrow and flat across tasks, models and domains. 

### The Stable Entropy Hypothesis:
We posit that any decoding algorithm that results in entropy of the model being "contained" with-in the stable entropy zone would result in more natural-sounding human-like generations. We use this hypothesis to analyze why deterministic decoding methods degenerate for open-ended generation and stochastic decoding methods do well. 

### Entropy-Aware Decoding:
We propose an entropy-aware decoding method that tries to keep the entropy of the model within the stable entropy zone while acting greedily most of the time. The details of the algorithm are available in the paper. The implementation is available [here](https://github.com/kushalarora/transformers/blob/main/src/transformers/generation_utils.py#L1894). 

We pass the stable entropy baseline and stable entropy zone to the algorithm by modeling them using a n-degree polynomial (n=2) and passing it to the algorithm.

## Requirements
To install requirements:

```setup
python -m venv ${HOME}/envs/ead
source ${HOME}/envs/ead/bin/activate

pip install -r requirements.txt
```

## Stable Entropy Analysis:
The stable entropy analysis is presented in ipython notebook [here](stable_entropy_analysis/stable_entropy_analysis.ipynb): 

```stable_entropy_analysis/stable_entropy_analysis.ipynb```

To generate data for correlation analysis, run:
```bash
./text_completion/wiki/generate_for_corr_analysis.sh
```

To compile the results in csv file, run:
```bash
./text_completion/compile_corr_analysis.py --directory data/wiki_rankgen/corr_analysis/gpt2_xl/
```

## Entropy-Aware Decoding Experiments:

Computing Coeffs for the stable entropy baseline (human_mean_coeffs) and the stable entropy zone (human_std_coeffs):
```bash
  # For text completion experiments:
  python entropy_aware_search/compute_human_entropy_coeffs.py --dataset data/wiki_rankgen/generated/orig.jsonl --model_name_or_path gpt2-xl --degree 2
  
  # For dialog experiments:
  python entropy_aware_search/compute_human_entropy_coeffs.py --dataset data/blended_skill_talk/generated/orig.jsonl --max_len 40 --model_name_or_path facebook/blenderbot-90M --is_seq2seq --degree 2
```

For generating using EAD, run:
```bash
# For text completion experiments:
./stable_entropy_hypothesis/text_completion/wiki/gpt2_v4_final_command.sh


# For dialog experiments:
./dialog/blended_skill_talk/generate_from_bb_ead_90M_final.sh

```
