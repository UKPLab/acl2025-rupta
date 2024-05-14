#!/bin/bash
#
#SBATCH --job-name=private-datasets
#SBATCH --output=/ukp-storage-1/yang/LLM_Anonymization/programming_runs/clss_train_out_reddit_openllama-3b.txt
#SBATCH --mail-user=yang@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1

source /ukp-storage-1/yang/reflexion/bin/activate
module purge
module load cuda/11.8
export WANDB_PROJECT=Privacy-NLP
export WANDB_LOG_MODEL=checkpoint
export WANDB_API_KEY=a3b3f7b7962a8b549c4635ee3a03944d554f1a10

python /ukp-storage-1/yang/LLM_Anonymization/programming_runs/run_classification.py --model_name_or_path FacebookAI/roberta-large --train_file /ukp-storage-1/yang/LLM_Anonymization/programming_runs/benchmarks/Reddit_synthetic/train.jsonl --validation_file /ukp-storage-1/yang/LLM_Anonymization/programming_runs/benchmarks/Reddit_synthetic/test.jsonl --shuffle_train_dataset --metric_name accuracy --text_column_name response --label_column_name label --do_train --do_eval --per_device_train_batch_size 16 --learning_rate 1e-5 --num_train_epochs 20 --output_dir /ukp-storage-1/yang/LLM_Anonymization/programming_runs/root/roberta-large_reddit_clss_b16_e20 --report_to wandb --run_name reddit_roberta-large_lr1e-5_B16_E20 --logging_steps 10 --eval_steps 20 --save_steps 20 --evaluation_strategy steps --load_best_model_at_end
