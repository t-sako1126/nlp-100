import os

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# 速度寄りの設定（精度影響は通常軽微）
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = "ch10/out/qwen-sst2-dpo"

MAX_LENGTH = 128
BETA = 0.1
USE_REFERENCE_MODEL = False


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
	model_name,
	torch_dtype="auto",
	device_map="auto",
)
model.config.use_cache = False
# 参照モデルの準備
ref_model = None
if USE_REFERENCE_MODEL:
	ref_model = AutoModelForCausalLM.from_pretrained(
		model_name,
		torch_dtype="auto",
		device_map="auto",
	)
	ref_model.config.use_cache = False
	ref_model.eval()
	for p in ref_model.parameters():
		p.requires_grad_(False)

def _prompt(example: dict) -> str:
	return f"""Decide if the sentiment is positive or negative.
Answer with only one word: positive or negative.

Sentence: {example['sentence']}
Answer:"""

# プロンプトとラベルをエンコードする関数
def _encode(prompt: str, answer: str) -> dict:
	# prompt 部分を -100 でマスクして、応答 tokens のみを学習対象にする
	prompt_text = tokenizer.apply_chat_template(
		[{"role": "user", "content": prompt}],
		tokenize=False,
		add_generation_prompt=True,
	)
	full_text = tokenizer.apply_chat_template(
		[
			{"role": "user", "content": prompt},
			{"role": "assistant", "content": answer},
		],
		tokenize=False,
		add_generation_prompt=False,
	)

	enc_prompt = tokenizer(
		prompt_text,
		add_special_tokens=False,
		truncation=True,
		max_length=MAX_LENGTH,
		padding=False,
	)
	enc_full = tokenizer(
		full_text,
		add_special_tokens=False,
		truncation=True,
		max_length=MAX_LENGTH,
		padding=False,
		return_attention_mask=True,
	)

	input_ids = enc_full["input_ids"]
	attention_mask = enc_full["attention_mask"]
	prompt_len = min(len(enc_prompt["input_ids"]), len(input_ids))
	labels = [-100] * prompt_len + input_ids[prompt_len:]
	return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def to_preference_pair(example: dict) -> dict:
	prompt = _prompt(example)
	chosen = "positive" if int(example["label"]) == 1 else "negative"
	rejected = "negative" if chosen == "positive" else "positive"

	c = _encode(prompt, chosen)
	r = _encode(prompt, rejected)

	return {
		"chosen_input_ids": c["input_ids"],
		"chosen_attention_mask": c["attention_mask"],
		"chosen_labels": c["labels"],
		"rejected_input_ids": r["input_ids"],
		"rejected_attention_mask": r["attention_mask"],
		"rejected_labels": r["labels"],
	}


class DPOCollator:
	def __init__(self, tokenizer: AutoTokenizer):
		self.tokenizer = tokenizer

	def _pad_labels(self, sequences: list[list[int]], max_len: int) -> torch.Tensor:
		return torch.tensor(
			[seq + [-100] * (max_len - len(seq)) for seq in sequences],
			dtype=torch.long,
		)

	def __call__(self, features: list[dict]) -> dict:
		chosen = [
			{"input_ids": f["chosen_input_ids"], "attention_mask": f["chosen_attention_mask"]}
			for f in features
		]
		rejected = [
			{"input_ids": f["rejected_input_ids"], "attention_mask": f["rejected_attention_mask"]}
			for f in features
		]

		c = self.tokenizer.pad(chosen, return_tensors="pt")
		r = self.tokenizer.pad(rejected, return_tensors="pt")

		c["labels"] = self._pad_labels([f["chosen_labels"] for f in features], c["input_ids"].shape[1])
		r["labels"] = self._pad_labels([f["rejected_labels"] for f in features], r["input_ids"].shape[1])

		return {
			"chosen_input_ids": c["input_ids"],
			"chosen_attention_mask": c["attention_mask"],
			"chosen_labels": c["labels"],
			"rejected_input_ids": r["input_ids"],
			"rejected_attention_mask": r["attention_mask"],
			"rejected_labels": r["labels"],
		}


def _sequence_logprob(m: AutoModelForCausalLM, input_ids, attention_mask, labels, no_grad: bool) -> torch.Tensor:
	ctx = torch.no_grad() if no_grad else torch.enable_grad()
	with ctx:
		logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
		shift_logits = logits[:, :-1, :]
		shift_labels = labels[:, 1:]

		mask = shift_labels != -100
		safe_labels = shift_labels.clone()
		safe_labels[~mask] = 0

		log_probs = F.log_softmax(shift_logits, dim=-1)
		tok_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
		tok_logp = tok_logp * mask
		return tok_logp.sum(dim=-1)


class DPOLikeTrainer(Trainer):
	def __init__(self, *args, beta: float = 0.1, ref_model=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.beta = beta
		self.ref_model = ref_model

	def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
		pi_c = _sequence_logprob(
			model,
			inputs["chosen_input_ids"],
			inputs["chosen_attention_mask"],
			inputs["chosen_labels"],
			no_grad=False,
		)
		pi_r = _sequence_logprob(
			model,
			inputs["rejected_input_ids"],
			inputs["rejected_attention_mask"],
			inputs["rejected_labels"],
			no_grad=False,
		)
		pi_logratios = pi_c - pi_r

		if self.ref_model is None:
			ref_logratios = torch.zeros_like(pi_logratios)
		else:
			ref_c = _sequence_logprob(
				self.ref_model,
				inputs["chosen_input_ids"],
				inputs["chosen_attention_mask"],
				inputs["chosen_labels"],
				no_grad=True,
			)
			ref_r = _sequence_logprob(
				self.ref_model,
				inputs["rejected_input_ids"],
				inputs["rejected_attention_mask"],
				inputs["rejected_labels"],
				no_grad=True,
			)
			ref_logratios = ref_c - ref_r

		logits = self.beta * (pi_logratios - ref_logratios)
		loss = -F.logsigmoid(logits).mean()
		return (loss, None) if return_outputs else loss


train_df = pd.read_csv("ch07/SST-2/train.tsv", sep="\t", header=0)
raw_train_dataset = Dataset.from_pandas(train_df)
train_dataset = raw_train_dataset.map(to_preference_pair, remove_columns=raw_train_dataset.column_names)

training_args = TrainingArguments(
	output_dir=output_dir,
	bf16=True,
	num_train_epochs=1,
	per_device_train_batch_size=8,
	gradient_accumulation_steps=8,
	learning_rate=5e-6,
	logging_steps=10,
	save_steps=1000,
	remove_unused_columns=False,
	report_to="none",
	optim="adamw_torch",
	dataloader_num_workers=2,
	dataloader_pin_memory=True,
)

trainer = DPOLikeTrainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	data_collator=DPOCollator(tokenizer),
	beta=BETA,
	ref_model=ref_model,
)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
