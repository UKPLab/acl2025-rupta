import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import find_all_linear_names, print_trainable_parameters

output_dir = "/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/root/kd-llama-7b/sft"
model_name = "/storage/ukp/shared/shared_model_weights/models--" + "llama-2-hf" + "/" + "7B"
train_data_path = '/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/benchmarks/Wiki_People/train_sampled3_sft.jsonl'
val_data_path = '/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/benchmarks/Wiki_People/val_sampled3_sft.jsonl'

train_dataset = load_dataset("json", data_files=train_data_path, split="train")
val_dataset = load_dataset("json", data_files=val_data_path, split="train")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  # torch_dtype=torch.float16
                                                  torch_dtype=torch.bfloat16,
                                                  quantization_config=bnb_config
                                                  )
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Change the LORA hyperparameters accordingly to fit your use case
peft_config = LoraConfig(
    r=128,
    lora_alpha=16,
    # target_modules=find_all_linear_names(base_model),
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"### Please anonymize the biography: {example['input'][i]}\n ### Anonymization result: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=15,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=5,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    run_name='llama-2-7b_lora-r-128-al-16_bz-4-acc-4_lr-2en4-ep-15',
    load_best_model_at_end=True,
    output_dir=output_dir,
    report_to=['wandb'],
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

trainer = SFTTrainer(
    base_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train()
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)