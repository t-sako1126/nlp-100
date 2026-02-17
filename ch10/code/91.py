import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval()

prompt = "The movie was full of"

# トークン化
input_ids = tokenizer.encode(prompt, return_tensors="pt")

def generate_text(sample, temperature):
  output = model.generate(
    input_ids,
    max_length=50,
    do_sample=sample,    
    temperature=temperature    
)
  return output
 
print("=== Greedy Search (temperature=1.0) ===")
output = generate_text(sample=False, temperature=1.0)
print(tokenizer.decode(output[0], skip_special_tokens=True))

print("\n=== Sampling (temperature=0.5) ===")
output = generate_text(sample=True, temperature=0.5)
print(tokenizer.decode(output[0], skip_special_tokens=True))

print("\n=== Sampling (temperature=1.5) ===")
output = generate_text(sample=True, temperature=1.5)
print(tokenizer.decode(output[0], skip_special_tokens=True))