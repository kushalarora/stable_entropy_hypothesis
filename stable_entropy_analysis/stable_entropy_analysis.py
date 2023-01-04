# %%
# Auto reload settings
#%%
# %%
# Imports

import sys

from entropy_aware_search.hf_utils import DataArguments, ModelArguments, get_tokenizer, get_model
from entropy_aware_search.utils import compute_average_across_sequences, process_datapoint
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from parlai.utils.strings import colorize
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import copy
pd.options.plotting.backend = "matplotlib"



# %%
# beautify graphs.

sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=11)    # legend fontsize
plt.rc('font', size=11)          # controls default text sizes

palatte=sns.color_palette('pastel')
palatte

# %%
# Plot mean and std average entropy

def plot_avg_entropies_mean_std(entropies, label=None, ax=None,  color='red', linewidth=1, std_dev=1):
    entropy_mean = np.ma.mean(entropies, axis=0)
    entropy_std = np.ma.std(entropies, axis=0)
    ax = sns.lineplot(y=entropy_mean, x=np.arange(len(entropy_mean)), 
                    ax=ax, label=label, color=color, linewidth=linewidth)
    # sns.lineplot(entropy_mean, ax=ax, label=label)

    ax.set_ylim(0,6)
    ax.fill_between(range(len(entropy_mean)), entropy_mean -  std_dev * entropy_std, 
                                                entropy_mean +  std_dev * entropy_std, alpha=0.1, color=color)
    # ax=ax.set_xticks(np.arange(len(entropy_mean)), step=100)


# %% [markdown]
# # Stable Entropy Baselines

# %%
NUM_SEQ=1000
MAX_LEN=128
WIDTH=5

# %%
# Text Completion data
wikipedia_text_completion = "/home/mila/a/arorakus/wdir/entropy_aware_search/data/wiki_rankgen/generated/orig.jsonl"
writing_prompt_completion = "/home/mila/a/arorakus/wdir/entropy_aware_search/data/writingPrompts/generated/orig.jsonl"
cc_news = "/home/mila/a/arorakus/wdir/entropy_aware_search/data/cc_news/generated/orig.jsonl"
pg19_completion = "/home/mila/a/arorakus/wdir/entropy_aware_search/data/pg19_rankgen/generated/orig.jsonl"
# Summarization

# %%
# GPT-2 XL Model
from transformers import AutoTokenizer, AutoModelForCausalLM
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")

# %% [markdown]
# ## Text Completion

# %%
wikipedia_dataframe = pd.read_json(wikipedia_text_completion, lines=True)\
                        .rename(columns={'prefix':'context', 
                                         'target': 'model_text'})
writing_prompt_dataframe = pd.read_json(writing_prompt_completion, lines=True)\
                        .rename(columns={'prefix':'context', 
                                         'target': 'model_text'})
cc_news_dataframe = pd.read_json(cc_news, lines=True)\
                        .rename(columns={'prefix':'context', 
                                         'target': 'model_text'})
pg19_dataframe = pd.read_json(pg19_completion, lines=True)\
                        .rename(columns={'prefix':'context', 
                                         'target': 'model_text'})

# %%
# Wikipedia and GPT-2 XL
print("Processing Wikipedia and GPT-2 XL")
dataframe, dataset_name = (wikipedia_dataframe, "Wikipedia")
model,tokenizer,model_name = (gpt2_model, gpt2_tokenizer, "GPT-2 XL")
_, human_ma_entropies = compute_average_across_sequences(dataframe, model, tokenizer, column_prefix='human_generated', width=WIDTH,  max_len=MAX_LEN, to_be_averaged='entropy_ma', num_seq=NUM_SEQ, cache=True)

fig, ax = plt.subplots(figsize=(3,2), tight_layout=True)
plot_avg_entropies_mean_std(human_ma_entropies, 
    label=f"{dataset_name} with {model_name}", ax=ax, 
    color='tab:green', linewidth=1.0)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
sns.move_legend(ax, loc="lower right", ncol=1)

# %%
# Writing Prompts and GPT-2 XL
print("Writing Prompts and GPT-2 XL")
dataframe, dataset_name = (writing_prompt_dataframe, "Writing Prompts")
model,tokenizer,model_name = (gpt2_model, gpt2_tokenizer, "GPT-2 XL")
_, human_ma_entropies = compute_average_across_sequences(dataframe, model, tokenizer, column_prefix='human_generated', width=WIDTH,  max_len=MAX_LEN, to_be_averaged='entropy_ma', num_seq=NUM_SEQ, cache=True)

fig, ax = plt.subplots(figsize=(3,2), tight_layout=True)
plot_avg_entropies_mean_std(human_ma_entropies, 
    label=f"{dataset_name} with {model_name}", ax=ax, 
    color='tab:green', linewidth=1.0)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
sns.move_legend(ax, loc="lower right", ncol=1)

# %%
# CC News and GPT-2 XL
print("CC News and GPT-2 XL")
dataframe, dataset_name = (cc_news_dataframe, "CC News")
model,tokenizer,model_name = (gpt2_model, gpt2_tokenizer, "GPT-2 XL")
_, human_ma_entropies = compute_average_across_sequences(dataframe, model, tokenizer, column_prefix='human_generated', width=WIDTH,  max_len=MAX_LEN, to_be_averaged='entropy_ma', num_seq=NUM_SEQ, cache=True)

fig, ax = plt.subplots(figsize=(3,2), tight_layout=True)
plot_avg_entropies_mean_std(human_ma_entropies, 
    label=f"{dataset_name} with {model_name}", ax=ax, 
    color='tab:green', linewidth=1.0)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
sns.move_legend(ax, loc="lower right", ncol=1)

# %%
# PG19 and GPT-2 XL
print("PG19 and GPT-2 XL")
dataframe, dataset_name = (pg19_dataframe, "PG19")
model,tokenizer,model_name = (gpt2_model, gpt2_tokenizer, "GPT-2 XL")
_, human_ma_entropies = compute_average_across_sequences(dataframe, model, tokenizer, column_prefix='human_generated', width=WIDTH,  max_len=MAX_LEN, to_be_averaged='entropy_ma', num_seq=NUM_SEQ, cache=True)

fig, ax = plt.subplots(figsize=(3,2), tight_layout=True)
plot_avg_entropies_mean_std(human_ma_entropies, 
    label=f"{dataset_name} with {model_name}", ax=ax, 
    color='tab:green', linewidth=1.0)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
sns.move_legend(ax, loc="lower right", ncol=1)

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
opt_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

# %%
# Wikipedia and OPT
print("Wikipedia and OPT")

dataframe, dataset_name = (wikipedia_dataframe, "Wikipedia")
model, tokenizer, model_name = (opt_model, opt_tokenizer, "OPT 1.3B")
_, human_ma_entropies = compute_average_across_sequences(dataframe, model, tokenizer, column_prefix='human_generated', width=WIDTH,  max_len=MAX_LEN, to_be_averaged='entropy_ma', num_seq=NUM_SEQ, cache=True)

fig, ax = plt.subplots(figsize=(3,2), tight_layout=True)
plot_avg_entropies_mean_std(human_ma_entropies, 
    label=f"{dataset_name} with {model_name}", ax=ax, 
    color='tab:green', linewidth=1.0)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
sns.move_legend(ax, loc="lower right", ncol=1)

# %%
# Writing Prompts and OPT
print("Writing Prompts and OPT")

dataframe, dataset_name = (writing_prompt_dataframe, "Writing Prompts")
model, tokenizer, model_name = (opt_model, opt_tokenizer, "OPT 1.3B")
_, human_ma_entropies = compute_average_across_sequences(dataframe, model, tokenizer, column_prefix='human_generated', width=WIDTH,  max_len=MAX_LEN, to_be_averaged='entropy_ma', num_seq=NUM_SEQ, cache=True)

fig, ax = plt.subplots(figsize=(3,2), tight_layout=True)
plot_avg_entropies_mean_std(human_ma_entropies, 
    label=f"{dataset_name} with {model_name}", ax=ax, 
    color='tab:green', linewidth=1.0)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
sns.move_legend(ax, loc="lower right", ncol=1)

# %%
# CC News and OPT
print("CC News and OPT")

dataframe, dataset_name = (cc_news_dataframe, "CC News")
model, tokenizer, model_name = (opt_model, opt_tokenizer, "OPT 1.3B")
_, human_ma_entropies = compute_average_across_sequences(dataframe, model, tokenizer, column_prefix='human_generated', width=WIDTH,  max_len=MAX_LEN, to_be_averaged='entropy_ma', num_seq=NUM_SEQ, cache=True)

fig, ax = plt.subplots(figsize=(3,2), tight_layout=True)
plot_avg_entropies_mean_std(human_ma_entropies, 
    label=f"{dataset_name} with {model_name}", ax=ax, 
    color='tab:green', linewidth=1.0)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
sns.move_legend(ax, loc="lower right", ncol=1)

# %%
# PG19 and GPT-2 XL
print("PG19 and GPT-2 XL")

dataframe, dataset_name = (pg19_dataframe, "PG19")
model,tokenizer,model_name = (opt_model, opt_tokenizer, "OPT 1.3B")
_, human_ma_entropies = compute_average_across_sequences(dataframe, model, tokenizer, column_prefix='human_generated', width=WIDTH,  max_len=MAX_LEN, to_be_averaged='entropy_ma', num_seq=NUM_SEQ, cache=True)

fig, ax = plt.subplots(figsize=(3,2), tight_layout=True)
plot_avg_entropies_mean_std(human_ma_entropies, 
    label=f"{dataset_name} with {model_name}", ax=ax, 
    color='tab:green', linewidth=1.0)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
sns.move_legend(ax, loc="lower right", ncol=1)

# %% [markdown]
# # Correlation Plots

# %%
correlation_data = pd.read_csv('/home/mila/a/arorakus/wdir/entropy_aware_search/data/wiki_rankgen/corr_analysis/gpt2_xl/compiled_results.csv', index_col=0)
correlation_data['Decoding Method'] = correlation_data['dataset'].str.extract("(top_k|temp|top_p|typical|eags|beam|greedy)")
correlation_data['Decoding Method'].str.replace("eags", "temperature")
correlation_data['Decoding Method'].str.replace("temp", "temperature")
correlation_data

# %%
fig, ax = plt.subplots(figsize=(5,4), tight_layout=True)

sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks

sns.scatterplot(data=correlation_data, x='lower_bound_violation_ratio', y='repeat_score@5', ax=ax, hue='Decoding Method')
# correlation_data.plot.scatter()
ax.set_xlabel("Entropy Lower Bound Violation Ratio")
ax.set_ylabel("Repeat Score@5")
sns.move_legend(ax, loc="lower right", ncol=2)


# %%
fig, ax = plt.subplots(figsize=(5,4), tight_layout=True)

sns.set_style('whitegrid') # darkgrid, white grid, dark, white and ticks

sns.scatterplot(data=correlation_data, x='entropy_violation_ratio', y='mauve', ax=ax, hue='Decoding Method')
# correlation_data.plot.scatter()
ax.set_xlabel("Entropy Violation Ratio")
ax.set_ylabel("Mauve Score")
sns.move_legend(ax, loc="upper right", ncol=2)


