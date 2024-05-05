#!/bin/bash
#
#SBATCH --job-name=llm_annonymization
#SBATCH --output=/ukp-storage-1/yang/LLM_Anonymization/programming_runs/llm_annom_privacy_evaluation_out_yi_34b_u_preview.txt
#SBATCH --mail-user=yang@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:v100:1

#export AZURE_OPENAI_ENDPOINT=https://azure-openai-ukp-west-us.openai.azure.com/
#export AZURE_OPENAI_API_KEY=228d82e8119943db9b990a25b94b6ef0
#export OPENAI_API_VERSION=2023-05-15
export PYTHONPATH=$PYTHONPATH:/ukp-storage-1/yang/LLM_Anonymization
source /ukp-storage-1/yang/reflexion/bin/activate
module purge
module load cuda/11.8

python /ukp-storage-1/yang/LLM_Anonymization/programming_runs/main.py --run_name evaluate_reflexion_yi_34b_u_preview --root_dir root --dataset_path /ukp-storage-1/yang/LLM_Anonymization/programming_runs/root/test_reflexion/evaluation/yi-34b_u_preview.jsonl --strategy test-acc --language text --pe_model gpt4-turbo-128k --pass_at_k 1 --max_iters 5 --verbose --p_threshold 10 --mem 3 --rag_data_path ./benchmarks/Wiki_People/All_data_for_retrieval.jsonl --rag_embed_cache_dir /home/ember/Desktop/work_space/Anonymization_Experiments/cache_emb --rag_num 5 --act_model meta-llama/Llama-2-70b-chat-hf --parser_model gpt-35-turbo-0301 --ue_model gpt4-turbo-128k
