import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import evaluate

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# GPU設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# パディングトークンがない場合は終端トークンを使用
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# モデル定義
class FrozenEmbedClassifier(nn.Module):
    def __init__(self, model_id, num_labels=2):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        # LLMのパラメータは更新しない
        for p in self.llm.parameters():
            p.requires_grad = False

        # 分類器の定義
        hidden = self.llm.config.hidden_size
        self.clf = nn.Linear(hidden, num_labels).to(
            device=self.llm.device, dtype=self.llm.dtype
        )
        self.loss_fn = nn.CrossEntropyLoss()
    # 順伝播
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        with torch.no_grad(): # LLM実行
            out = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # 最後の隠れ状態の平均を特徴量とする
            last_hidden = out.hidden_states[-1] 
            emb = last_hidden.mean(dim=1) # 平均
        logits = self.clf(emb)

        loss = None
        # ラベルがある場合は損失を計算
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

# モデルとデータコラレータの準備
model = FrozenEmbedClassifier(MODEL_ID)
data_collator = DataCollatorWithPadding(tokenizer)

# データ読み込み
train_df = pd.read_csv("ch07/SST-2/train.tsv", sep="\t", header=0)
dev_df   = pd.read_csv("ch07/SST-2/dev.tsv",   sep="\t", header=0)

# Trainer用にカラム名を変える
train_df = train_df.rename(columns={"label": "labels"})
dev_df   = dev_df.rename(columns={"label": "labels"})

# Dataset化
train_ds = Dataset.from_pandas(train_df[["sentence", "labels"]])
dev_ds   = Dataset.from_pandas(dev_df[["sentence", "labels"]])

def preprocess(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=64)

train_ds = train_ds.map(preprocess, batched=True)
dev_ds   = dev_ds.map(preprocess, batched=True)

# テンソル形式
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dev_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 評価
acc = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    return acc.compute(predictions=preds, references=eval_pred.label_ids)

training_args = TrainingArguments(
    output_dir="ch10/out/97",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=256,
    learning_rate=1e-3,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    num_train_epochs=1,
    save_strategy="no",
    fp16=False,  
    report_to="none",
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
eval_result = trainer.evaluate()
print(f"dev accuracy: {eval_result['eval_accuracy']:.4f}")