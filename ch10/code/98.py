import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# データセットの準備と整形・pandasのDataFrameをDatasetオブジェクトに変換
train_df = pd.read_csv("ch07/SST-2/train.tsv", sep="\t", header=0)
raw_train_dataset = Dataset.from_pandas(train_df)

def prompt_template(example):
    # 指示と入力文を組み合わせたプロンプトの作成
    prompt = f"""Decide if the sentiment is positive or negative.
Answer with only one word: positive or negative.

Sentence: {example['sentence']}
Answer:"""

    # ラベルをテキストに変換
    label_text = "positive" if example['label'] == 1 else "negative"
    
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label_text}
        ]
    }

# データセットの整形
formatted_dataset = raw_train_dataset.map(prompt_template)


# トークン化関数の定義
def tokenize_function(example):
# チャット形式のテキストを作成
    chat_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
# トークン化
    enc = tokenizer(
        chat_text,
        truncation=True,
        max_length=256,
        padding=False,
        add_special_tokens=False,
        return_attention_mask=True,
    )

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    remove_columns=formatted_dataset.column_names
)

training_args = TrainingArguments(
    output_dir="ch10/out/qwen-sst2",
    bf16=True, 
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    logging_steps=10,
    save_steps=1000,

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("ch10/out/qwen-sst2")
tokenizer.save_pretrained("ch10/out/qwen-sst2")
