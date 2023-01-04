#!/bin/bash
set -e

for filename in data/wiki_rankgen/corr_analysis/gpt2_xl/*.jsonl;
do
    if [ -f "${filename}" ] && [ $(($(wc -l < "${filename}") > 5000)) -eq 1 ] && [ ! -f "${filename}.score" ];
    then
        echo ${filename}
        sbatch -t 1:00:00 ./launcher_basic.sh python text_completion/score_generations.py --dataset ${filename}
    fi
done