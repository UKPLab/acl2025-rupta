import os
import torch
import json

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from datasets import load_dataset
import datasets
from trl import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from utils import find_all_linear_names, print_trainable_parameters, get_max_length


model_name = "llama-3"
model_version = "8B-Instruct"
# model_name = "microsoft--phi-3"
# model_version = "mini-4k-instruct"
# model_name = "google--flan-t5-large--hf"
# model_version = ""
model_path = os.path.join("/storage/ukp/shared/shared_model_weights/models--" + model_name, model_version)
train_data_path = '/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/benchmarks/Wiki_People/train_sampled3_dpo.jsonl'
val_data_path = '/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/benchmarks/Wiki_People/val_sampled3_dpo.jsonl'
data_format = 'Dialog' #or Dialog Instruction
lora_r = 32
lora_alpha = 64
lr = 1e-4
epoch = 7
batch_size = 4
batch_acc = 4
run_name = f"dpo_{model_name}-{model_version}_{data_format}_lr-{lr}_ep-{epoch}_bz-{batch_size}-acc-{batch_acc}_lora-r-{lora_r}-ap-{lora_alpha}"
output_dir = "/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/root/knowledge_distillation/dpo/" + run_name
sft_model_name = "/mnt/beegfs/work/yang/LLM_Anonymization/programming_runs/root/knowledge_distillation/sft/" + "sft_llama-3-8B-Instruct_Dialog_lr-0.0001_ep-7_bz-4-acc-4_lora-r-32-ap-64" + '/' + "final_merged_checkpoint"

with open(train_data_path) as f:
    train_data_raw = []
    raw = f.readlines()
    for r in raw:
        train_data_raw.append(json.loads(r))
with open(val_data_path) as f:
    val_data_raw = []
    raw = f.readlines()
    for r in raw:
        val_data_raw.append(json.loads(r))
train_dataset = datasets.Dataset.from_list(train_data_raw)
val_dataset = datasets.Dataset.from_list(val_data_raw)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

if model_name != "google--flan-t5-large--hf":
    model = AutoModelForCausalLM.from_pretrained(sft_model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    model_ref = AutoModelForCausalLM.from_pretrained(sft_model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(sft_model_name, torch_dtype=torch.bfloat16,
                                                 quantization_config=bnb_config)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    model_ref = AutoModelForSeq2SeqLM.from_pretrained(sft_model_name, torch_dtype=torch.bfloat16,
                                                     quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(sft_model_name)
tokenizer.pad_token = tokenizer.eos_token
max_length = get_max_length(model)


def return_prompt_and_responses_seq2seq(samples):
    return {
        "prompt": [
            f"Please anonymize the biography:\n\n{input}\n\nAnonymization result:"
            for input in samples["original"]
        ],
        "chosen": samples["chosen"],
        "rejected": samples["reject"],
    }

def return_prompt_and_responses_dec(samples):
    return {
        "prompt": [tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful text anonymizer."},
            {"role": "user", "content": f"Please anonymize the following biography:\n{input}"}
        ], tokenize=False) for input in samples["original"]
        ],
        "chosen": [tokenizer.apply_chat_template([
            {"role": "assistant", "content": f"Anonymization result:\n{chosen}"}
        ], tokenize=False) for chosen in samples["chosen"]
        ],
        "rejected": [tokenizer.apply_chat_template([
            {"role": "assistant", "content": f"Anonymization result:\n{reject}"}
        ], tokenize=False) for reject in samples["reject"]
        ],
    }

original_columns = train_dataset.column_names
train_dataset = train_dataset.map(
    return_prompt_and_responses_seq2seq if model_name == "google--flan-t5-large--hf" else return_prompt_and_responses_dec,
    batched=True,
    remove_columns=original_columns
)
val_dataset = val_dataset.map(
    return_prompt_and_responses_seq2seq if model_name == "google--flan-t5-large--hf" else return_prompt_and_responses_dec,
    batched=True,
    remove_columns=original_columns
)

training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    save_strategy='epoch',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=batch_acc,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=epoch,
    # save_steps=200,
    # eval_steps=200,
    learning_rate=lr,
    run_name=run_name,
    load_best_model_at_end=True,
    report_to=['wandb'],
    bf16=True,
    save_total_limit=5,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    remove_unused_columns=False
)

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=find_all_linear_names(model),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" if model_name != 'google--flan-t5-large--hf' else "SEQ_2_SEQ_LM",
)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

if model_name != 'google--flan-t5-large--hf':
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=2048,
    )
else:
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_prompt_length=max_length,
        max_target_length=max_length,
        max_length=max_length * 2,
        truncation_mode='keep_start',
        is_encoder_decoder=True
    )


dpo_trainer.train()
dpo_trainer.save_model(output_dir)


output_dir = os.path.join(output_dir, "final_checkpoint")
dpo_trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)