#!/bin/bash
set -ex

directory="data/blended_skill_talk/generated/BB_1B/"
mkdir -p directory

# Greedy
filename=${directory}/greedy.jsonl
if [[ ! -f "${filename}" ]] || [[ $(($(wc -l < ${filename}) < 5000)) -eq 1 ]];
then
    echo "Running Greedy."
    JOBID1=$(sbatch -t 30:00  --gres=gpu:2g.20gb:1 --parsable  ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16 --model_name_or_path facebook/blenderbot-1B-distill --batch_size 64 --output_filename ${filename}); 
    echo $JOBID1
    sbatch -t 45:00 --gres=gpu:2g.20gb:1  --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
fi

# Greed 3-gram beam block
filename=${directory}/greedy_3gram_beam_block.jsonl
if [[ ! -f "${filename}" ]] || [[ $(($(wc -l < ${filename}) < 5000)) -eq 1 ]];
then
    echo "Running Greedy 3 gram beam block."
    JOBID1=$(sbatch -t 30:00  --gres=gpu:2g.20gb:1 --parsable  ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16 --model_name_or_path facebook/blenderbot-1B-distill --batch_size 64 --output_filename ${filename} --no_repeat_ngram_size 3); 
    echo $JOBID1
    sbatch -t 45:00 --gres=gpu:2g.20gb:1  --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
fi

# Beam
for num_beams in 5; 
do 
        filename=${directory}/beam_${num_beams}.jsonl
        echo "Running Beam Search: ${num_beams}."
        JOBID1=$(sbatch -t 2:00:00 --parsable --gres=gpu:2g.20gb:1  ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16   --model_name_or_path facebook/blenderbot-1B-distill --output_filename ${filename} --num_beams ${num_beams} --batch_size $((64/${num_beams})));
        echo $JOBID1
        sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;

        echo "Running Beam Search 3-gram beam block: ${num_beams}."
        filename=${directory}/beam_${num_beams}_3gram_beam_block.jsonl
        JOBID1=$(sbatch -t 2:00:00 --parsable --gres=gpu:2g.20gb:1  ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16   --model_name_or_path facebook/blenderbot-1B-distill --output_filename ${filename} --num_beams ${num_beams} --batch_size $((64/${num_beams}))  --no_repeat_ngram_size 3);
        echo $JOBID1
        sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
done

for run in 1; do
    for p in 0.9; 
    # for p in 0.15 0.4; 
    do
        filename="${directory}/top_p_${p}_run_${run}.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running Nucleus Sampling: run=${run} || top-p=${p}."
            JOBID1=$(sbatch -t 30:00  --gres=gpu:2g.20gb:1 --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_filename ${filename}  --p ${p} --do_sample  --seed ${seed}  --batch_size 64);
            echo $JOBID1
            sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
        fi
    done
# done
    for k in 30; 
    do
        filename="${directory}/top_k_${k}_run_${run}.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running Top-k Sampling: run=${run} || top-k=${k}."
            JOBID1=$(sbatch -t 30:00  --gres=gpu:2g.20gb:1 --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_filename ${filename}  --k ${k} --do_sample  --seed ${seed}  --batch_size 64);
            echo $JOBID1
            sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
        fi
    done

    # for tau in 0.2 0.9; 
    # do
    #     filename="${directory}/typical_p_${tau}_run_${run}.jsonl"
    #     seed=$((1 + RANDOM % 1000))
    #     if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
    #     then
    #         echo "Running Typical Sampling: run=${run} || tau=${tau}."
    #         JOBID1=$(sbatch -t 30:00  --gres=gpu:2g.20gb:1 --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_filename ${filename}  --typical_p ${tau} --do_sample --seed ${seed} --batch_size 64);
    #         echo $JOBID1
    #         sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
    #     fi
    # done

    for std_dev in 0.25 0.5 0.75 1; 
    do
        filename="${directory}/eas_std_dev_${std_dev}_run_${run}_degree_1.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running EAS: run=${run} || std_dev=${std_dev}."
            JOBID1=$(sbatch -t 30:00  --gres=gpu:2g.20gb:1 --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_filename ${filename}  --seed ${seed} --batch_size 64 --ea_human_mean_coeffs -0.00102 2.7048 --ea_human_std_coeffs 0.00735 0.66831 --ea_human_entropy_std_band ${std_dev} --entropy_aware_sampling  --do_sample);
            echo $JOBID1
            sbatch -t 1:00:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
        fi

        filename="${directory}/ead_std_dev_${std_dev}_k_30_run_${run}_degree_1.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running EAD: run=${run} || std_dev=${std_dev}."
            JOBID1=$(sbatch -t 45:00  --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_filename ${filename}  --seed ${seed} --batch_size 64 --ea_human_mean_coeffs -0.00102 2.7048 --ea_human_std_coeffs 0.00735 0.66831 --ea_human_entropy_std_band ${std_dev} --k 30);
            echo $JOBID1
            sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
        fi

        filename="${directory}/ead_std_dev_${std_dev}_k_30_run_${run}_degree_1_ea_donot_intervene_for_lower_bound.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running EAD: run=${run} || std_dev=${std_dev}."
            JOBID1=$(sbatch -t 45:00  --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_filename ${filename}  --seed ${seed} --batch_size 64 --ea_human_mean_coeffs -0.00102 2.7048 --ea_human_std_coeffs 0.00735 0.66831 --ea_human_entropy_std_band ${std_dev} --k 30  --ea_donot_intervene_for_lower_bound);
            echo $JOBID1
            sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
        fi

        filename="${directory}/ead_std_dev_${std_dev}_p_0.9_run_${run}_degree_1.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running EAD: run=${run} || std_dev=${std_dev}."
            JOBID1=$(sbatch -t 45:00  --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_filename ${filename}  --seed ${seed} --batch_size 64 --ea_human_mean_coeffs -0.00102 2.7048 --ea_human_std_coeffs 0.00735 0.66831 --ea_human_entropy_std_band ${std_dev} --p 0.9);
            echo $JOBID1
            sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
        fi

        filename="${directory}/ead_std_dev_${std_dev}_p_0.9_run_${run}_degree_1_ea_donot_intervene_for_lower_bound.jsonl"
        seed=$((1 + RANDOM % 1000))
        if [ ! -f "${filename}" ] || [ $(($(wc -l < "${filename}") < 5000)) -eq 1 ];
        then
            echo "Running EAD: run=${run} || std_dev=${std_dev}."
            JOBID1=$(sbatch -t 45:00  --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_filename ${filename}  --seed ${seed} --batch_size 64 --ea_human_mean_coeffs -0.00102 2.7048 --ea_human_std_coeffs 0.00735 0.66831 --ea_human_entropy_std_band ${std_dev} --p 0.9  --ea_donot_intervene_for_lower_bound);
            echo $JOBID1
            sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py --model_name_or_path facebook/blenderbot-1B-distill  --dataset ${filename} --is_seq2seq;
        fi
    done
done

for std_dev in 0.25 0.5 0.75 1; 
do
    filename="${directory}/ead_std_dev_${std_dev}_k_30_run_${run}_degree_1_ea_donot_intervene_for_upper_bound.jsonl"
    echo "Running EAD: run=${run} || std_dev=${std_dev}."
    JOBID1=$(sbatch -t 45:00 --parsable ./launcher_basic.sh python dialog/blended_skill_talk/generate_from_bb.py --fp16  --model_name_or_path facebook/blenderbot-1B-distill  --output_filename ${filename}  --seed ${seed} --batch_size 64 --ea_human_mean_coeffs -0.00102 2.7048 --ea_human_std_coeffs 0.00735 0.66831 --ea_human_entropy_std_band ${std_dev} --ea_donot_intervene_for_upper_bound);
    echo $JOBID1
    sbatch -t 45:00 --gres=gpu:2g.20gb:1 --parsable --dependency=afterok:${JOBID1} ./launcher_basic.sh python dialog/score_generations.py  --is_seq2seq --model_name_or_path facebook/blenderbot-1B-distill --dataset ${filename};
done