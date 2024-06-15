import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM

torch.cuda.empty_cache()

model_name = "llama-3"
model_version = "8B-Instruct"
# model_name = "microsoft--phi-3"
# model_version = "mini-4k-instruct"
# model_name = "google--flan-t5-large--hf"
# model_version = ""
model_path = os.path.join("/storage/ukp/shared/shared_model_weights/models--" + model_name, model_version)
adapter_dir = "/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/root/knowledge_distillation/sft/" + "sft_llama-3-8B-Instruct_Dialog_lr-0.0001_ep-7_bz-4-acc-4_lora-r-32-ap-64" + '/checkpoint-847'
output_dir = "/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/root/knowledge_distillation/sft/" + "sft_llama-3-8B-Instruct_Dialog_lr-0.0001_ep-7_bz-4-acc-4_lora-r-32-ap-64"

if model_name != "google--flan-t5-large--hf":
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, device_map="cpu", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
else:
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(adapter_dir, device_map="cpu", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    # peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(model_path,
    #                                                         torch_dtype=torch.bfloat16)
    #
    # model = PeftModel.from_pretrained(peft_model_base,
    #                                   adapter_dir,
    #                                   torch_dtype=torch.bfloat16,
    #                                   is_trainable=False)

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
tokenizer.save_pretrained(output_merged_dir)