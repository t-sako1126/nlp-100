# 90.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"
MODEL = "gpt2"
PROMPT = "The movie was full of"
K = 10

# トークナイザとモデルの読み込み
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
model.to(device)

# プロンプトをトークン化してモデルに入力
x = tok(PROMPT, return_tensors="pt")
x = {k: v.to(device) for k, v in x.items()}
ids = x["input_ids"][0].tolist()
print("TOKENS:", [tok.decode([i]) for i in ids]) # トークン化されたプロンプトを表示

# モデルの出力から次のトークンの確率を計算して表示
with torch.no_grad():
    logits = model(**x).logits[0, -1] 
    probs = torch.softmax(logits, dim=-1)
    p, i = torch.topk(probs, K)

for pp, ii in zip(p.tolist(), i.tolist()):
    print(f"{pp:.6f}\t{repr(tok.decode([ii]))}")
