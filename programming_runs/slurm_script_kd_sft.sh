#!/bin/bash
#
#SBATCH --job-name=private-datasets
#SBATCH --output=/ukp-storage-1/yang/LLM_Anonymization/programming_runs/kd_sft_generate_flan-t5.txt
#SBATCH --mail-user=yang@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:v100:1

source /ukp-storage-1/yang/dpo_h/bin/activate
module purge
module load cuda/11.8
export WANDB_PROJECT=Privacy-NLP
#export WANDB_LOG_MODEL=checkpoint
export WANDB_API_KEY=a3b3f7b7962a8b549c4635ee3a03944d554f1a10

#python /ukp-storage-1/yang/LLM_Anonymization/programming_runs/knowledge_distillation/sft_trainer.py
#python /ukp-storage-1/yang/LLM_Anonymization/programming_runs/knowledge_distillation/merge_peft_adapters.py
#python /ukp-storage-1/yang/LLM_Anonymization/programming_runs/knowledge_distillation/dpo_trainer.py
python /ukp-storage-1/yang/LLM_Anonymization/programming_runs/knowledge_distillation/generate.py
