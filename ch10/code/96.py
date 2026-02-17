import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

dev_df = pd.read_csv("ch07/SST-2/dev.tsv", sep="\t", header=0)

preds = []

for _, row in dev_df.iterrows():
    sentence = row["sentence"]

    prompt = f"""Decide if the sentiment is positive or negative.
             Answer with only one word: positive or negative.

             Sentence: {sentence}
             Answer:"""

    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device) # トークン化
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False) # 生成
    gen = out[0][inputs.input_ids.shape[1]:] # 生成部分の切り出し
    pred_text = tokenizer.decode(gen, skip_special_tokens=True).lower() # 生成テキストのデコード

    pred = 1 if "positive" in pred_text else 0 # 予測の変換

    preds.append(pred)

dev_df["pred"] = preds

acc = (dev_df["pred"] == dev_df["label"]).mean()

print("accuracy:", acc)
