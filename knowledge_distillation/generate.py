import torch
import json
import os
from peft import AutoPeftModelForCausalLM, PeftModel, AutoPeftModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import tqdm
from utils import get_max_length

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_name ="./merged_peft/final_merged_checkpoint"
# model_name = "llama-3"
# model_name = "microsoft--phi-3"
model_name = "google--flan-t5-large--hf"
# model_version = "8B-Instruct"
# model_version = "mini-4k-instruct"
model_version = ""
model_path = os.path.join("/storage/ukp/shared/shared_model_weights/models--" + model_name, model_version)
adapter_path = "/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/root/knowledge_distillation/sft/sft_google--flan-t5-large--hf-_Original_lr-0.0002_ep-15_bz-8-acc-4_lora-r-128-ap-256/final_checkpoint"
output_path = "/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/root/knowledge_distillation/sft/sft_google--flan-t5-large--hf-_Original_lr-0.0002_ep-15_bz-8-acc-4_lora-r-128-ap-256/generation_results_final_checkpoint_sample.jsonl"
test_data_path = '/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/benchmarks/Wiki_People/test_sampled3.jsonl'
with open(test_data_path, 'r') as f:
    raw = f.readlines()
    test_data = []
    for r in raw:
        test_data.append(json.loads(r))
# adapter_path = "./dpo_results/final_checkpoint"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if model_name != "google--flan-t5-large--hf":
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

if model_name != 'google--flan-t5-large--hf':
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    ).to(DEV)
    messages = [
        {"role": "system", "content": "You are a helpful text anonymizer."},
        {"role": "user", "content": "Please anonymize the following biography:\n{biography}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
else:
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    ).to(DEV)
    # peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(model_path,
    #                                                         torch_dtype=torch.bfloat16)

    # model = PeftModel.from_pretrained(peft_model_base,
    #                                   adapter_path,
    #                                   torch_dtype=torch.bfloat16,
    #                                   is_trainable=False).to(DEV)
    prompt = "Please anonymize the biography:\n\n{biography}\n\nAnonymization result: "

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     load_in_4bit=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# inputs = tokenizer.encode("An AI tool that corrects and rephrase user text grammar errors delimited by triple backticks to standard English.\n### Input: ```here is how write for loop in js```\n### Output:", return_tensors="pt").to(DEV)
anonymization_results = []
# start_idx = list(range(0, len(test_data[:4]), 4))
# end_idx = list(range(4, len(test_data[:4]) + 4, 4))
max_length = get_max_length(model)
# for s_id, e_id in tqdm.tqdm(zip(start_idx, end_idx)):
for d in tqdm.tqdm(test_data):
    # texts = test_data[s_id:e_id] if e_id < len(test_data) else test_data[s_id:]
    # inputs = tokenizer([prompt.format(biography=d['text']) + "Anonymization result:\n" for d in texts], padding=True, truncation=True, return_tensors="pt").to(DEV)
    if model_name != 'google--flan-t5-large--hf':
        inputs = tokenizer.encode(prompt.format(biography=d['text']) + "Anonymization result:\n", return_tensors="pt").to(DEV)
    else:
        inputs = tokenizer.encode(prompt.format(biography=d['text']), truncation=True, max_length=max_length, return_tensors="pt").to(DEV)
    outputs = model.generate(
        input_ids=inputs,
                             do_sample=True,
                             temperature=0.1,
                             top_p=0.95,
                             top_k=40,
                             max_new_tokens=1024,
                             repetition_penalty=1.5,
                             # pad_token_id=128001,
                             # eos_token_id=tokenizer.encode('<|eot_id|>')[0]
                             )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
    # ]
    temp = {}
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    temp['original'] = result
    if model_name != 'google--flan-t5-large--hf':
        result_pure = result.replace(prompt.format(biography=d['text']), '')
        temp['pure'] = result_pure
    anonymization_results.append(temp)

with open(output_path, 'w') as f:
    for t in anonymization_results:
        f.write(json.dumps(t) + '\n')



