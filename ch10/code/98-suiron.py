import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 学習済みモデルのパス =====
model_path = "ch10/out/qwen-sst2"

# ===== モデルロード =====
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

# ===== 推論したい文章 =====
sentence = "This movie was absolutely fantastic!"

# 学習時と同じプロンプト形式を作る（超重要）
prompt = f"""Decide if the sentiment is positive or negative.
Answer with only one word: positive or negative.

Sentence: {sentence}
Answer:"""

messages = [
    {"role": "user", "content": prompt}
]

# ===== chat template を適用 =====
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # ← 推論では True
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# ===== 生成 =====
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,   # 分類なので deterministic 推奨
    )

# ===== デコード =====
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
