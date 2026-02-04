import os
import numpy as np
import pandas as pd
import torch
import evaluate

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# GPU設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    torch.cuda.set_device(0)

MODEL_ID = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

data_collator = DataCollatorWithPadding(tokenizer)

# データ読み込み
train_df = pd.read_csv("ch07/SST-2/train.tsv", sep="\t", header=0)
dev_df   = pd.read_csv("ch07/SST-2/dev.tsv",   sep="\t", header=0)

# Trainer用にカラム名を変える
train_df = train_df.rename(columns={"label": "labels"})
dev_df   = dev_df.rename(columns={"label": "labels"})

# Datasetオブジェクトに変換
train_ds = Dataset.from_pandas(train_df[["sentence", "labels"]])
dev_ds   = Dataset.from_pandas(dev_df[["sentence", "labels"]])


def preprocess(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=64)

train_ds = train_ds.map(preprocess, batched=True)
dev_ds   = dev_ds.map(preprocess, batched=True)

# テンソル形式に変換
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dev_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

acc = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return acc.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="ch09/out/87",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=256,
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,        
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

eval_result = trainer.evaluate(dev_ds)
print(f"dev accuracy: {eval_result['eval_accuracy']:.4f}")
