#!/bin/bash
#
#SBATCH --job-name=llm_annonymization
#SBATCH --output=/ukp-storage-1/yang/LLM_Anonymization/programming_runs/llm_annom_out_utility_gpt4tb_train_new_465-600.txt
#SBATCH --mail-user=yang@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:v100:1

#export OPENAI_API_BASE=https://azure-openai-ukp-004.openai.azure.com/
#export OPENAI_API_KEY=9443b9b3e9d44a648822744086b078dd
#export AZURE_OPENAI_ENDPOINT=https://azure-openai-ukp-004.openai.azure.com/
#export AZURE_OPENAI_API_KEY=9443b9b3e9d44a648822744086b078dd
#export OPENAI_API_VERSION=2023-05-15
export PYTHONPATH=$PYTHONPATH:/ukp-storage-1/yang/LLM_Anonymization
source /ukp-storage-1/yang/reflexion/bin/activate
module purge
module load cuda/11.8

python /ukp-storage-1/yang/LLM_Anonymization/programming_runs/main.py --run_name test_reflexion_train_465-600_new --root_dir /ukp-storage-1/yang/LLM_Anonymization/programming_runs/root --dataset_path /ukp-storage-1/yang/LLM_Anonymization/programming_runs/benchmarks/Wiki_People/train_sampled3.jsonl --strategy reflexion --language wiki --pass_at_k 1 --max_iters 5 --verbose --p_threshold 10 --mem 3 --rag_data_path ./benchmarks/Wiki_People/All_data_for_retrieval.jsonl --rag_embed_cache_dir /home/ember/Desktop/work_space/Anonymization_Experiments/cache_emb --rag_num 5 --pe_model gpt4-turbo-128k --ue_model gpt4-turbo-128k --act_model gpt4-turbo-128k --parser_model gpt4-turbo-128k
