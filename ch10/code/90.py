import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval()

prompt = "The movie was full of"

# トークン化
inputs = tokenizer(prompt, return_tensors="pt")

#promptのトークンを出力
print("トークンID:", inputs["input_ids"].tolist()[0])
print("トークン:", tokenizer.decode(inputs["input_ids"][0]))

# モデルの出力を取得
with torch.no_grad():
    outputs = model(**inputs) 

# 最後のトークンのロジットを取得
logits = outputs.logits[:, -1, :]

# 確率に変換
probs = torch.softmax(logits, dim=-1)

topk = torch.topk(probs, k=10)
for token_id, prob in zip(topk.indices[0], topk.values[0]):
    token = tokenizer.decode([token_id])
    print(f"{token!r} : {prob.item():.4f}")
