#!/bin/bash
#
#SBATCH --job-name=llm_annonymization
#SBATCH --output=/ukp-storage-1/yang/LLM_Anonymization/programming_runs/llm_annom_evaluation_out_presidio_roberta.txt
#SBATCH --mail-user=yang@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:v100:1

export OPENAI_API_BASE=https://azure-openai-ukp-004.openai.azure.com/
export OPENAI_API_KEY=9443b9b3e9d44a648822744086b078dd
export PYTHONPATH=$PYTHONPATH:/ukp-storage-1/yang/LLM_Anonymization
source /ukp-storage-1/yang/reflexion/bin/activate
module purge
module load cuda/11.8

python /ukp-storage-1/yang/LLM_Anonymization/programming_runs/main.py --run_name evaluate_reflexion --root_dir /ukp-storage-1/yang/LLM_Anonymization/programming_runs/root --dataset_path /ukp-storage-1/yang/LLM_Anonymization/programming_runs/root/test_reflexion/evaluation/presidio_roberta.jsonl --strategy test-acc --language text --model gpt-4 --pass_at_k 1 --max_iters 5 --verbose --p_threshold 10 --mem 3
