import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval()

prompt = "The movie was full of"

# トークン化
input_ids = tokenizer.encode(prompt, return_tensors="pt")
prompt_len = input_ids.shape[1]

output_ids = model.generate(
    input_ids,
    max_new_tokens=10,
    do_sample=True,
    top_k=50
)
# 生成されたトークンをデコードして表示
generated_text = tokenizer.decode(output_ids[0])
print("Generated:", generated_text)

# 生成されたトークンの確率を計算
with torch.no_grad():
    outputs = model(output_ids)

logits = outputs.logits

# softmaxで確率化
probs = torch.softmax(logits, dim=-1)

tokens = output_ids[0]

print("\nToken probabilities:")
for i in range(prompt_len, len(tokens)):
    token_id = tokens[i]
    token_str = tokenizer.decode(token_id)

    # 1つ前の位置の確率分布を見る
    prob = probs[0, i-1, token_id]

    print(token_str, float(prob))
