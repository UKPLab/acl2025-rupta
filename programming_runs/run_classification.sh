export WANDB_LOG_MODEL=checkpoint
export WANDB_PROJECT=Privacy-NLP
export WANDB_API_KEY=a3b3f7b7962a8b549c4635ee3a03944d554f1a10
python run_classification.py \
	--model_name_or_path google-bert/bert-base-uncased \
	--train_file benchmarks/Wiki_People/train_sampled.jsonl \
	--validation_file benchmarks/Wiki_People/val_sampled.jsonl \
	--test_file benchmarks/Wiki_People/test_sampled.jsonl \
	--shuffle_train_dataset \
	--metric_name accuracy \
	--text_column_name text \
	--label_column_name label \
	--do_train \
	--do_eval \
	--do_predict \
	--max_seq_length 512 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--lr_scheduler_type linear \
	--warmup_steps 100 \
	--num_train_epochs 20 \
	--output_dir root/bert_cls_lr2e-5_B32_li-w100 \
	--report_to wandb \
	--run_name lr2e-5_B32_li-w100 \
	--logging_steps 10 \
	--eval_steps 100 \
	--save_steps 100 \
	--load_best_model_at_end \
	--metric_for_best_model accuracy \
	--evaluation_strategy steps
