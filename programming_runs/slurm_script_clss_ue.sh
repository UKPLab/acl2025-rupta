#!/bin/bash
#
#SBATCH --job-name=private-datasets
#SBATCH --output=/ukp-storage-1/yang/LLM_Anonymization/programming_runs/clss_eval_out.txt
#SBATCH --mail-user=yang@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:v100:1

source /ukp-storage-1/yang/reflexion/bin/activate
module purge
module load cuda/11.8
export WANDB_PROJECT=Privacy-NLP
export WANDB_LOG_MODEL=checkpoint
export WANDB_API_KEY=a3b3f7b7962a8b549c4635ee3a03944d554f1a10

python /ukp-storage-1/yang/LLM_Anonymization/programming_runs/run_classification.py --model_name_or_path /mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/root/bert_cls_sampled3 --train_file /mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/root/test_reflexion/evaluation/presidio_stanza.jsonl --validation_file /mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/root/test_reflexion/evaluation/presidio_stanza.jsonl --shuffle_train_dataset --metric_name accuracy --text_column_name anonymized_text --label_column_name label --do_eval --max_seq_length 512 --per_device_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 20 --output_dir /ukp-storage-1/yang/LLM_Anonymization/programming_runs/root/bert_cls_sampled3/evaluation --report_to wandb --run_name lr2e-5_B32 --logging_steps 10 --eval_steps 100 --save_steps 100 --load_best_model_at_end --evaluation_strategy steps
