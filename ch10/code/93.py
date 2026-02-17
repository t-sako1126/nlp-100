import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval()

model.eval()

texts = [
 "The movie was full of surprises",
 "The movies were full of surprises",
 "The movie were full of surprises",
 "The movies was full of surprises" 
]

for text in texts:
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    
    with torch.no_grad():
      outputs = model(input_ids, labels=input_ids)
      loss = outputs.loss

      ppl = math.exp(loss.item())
      print(f"Text: {text}")
      print("Perplexity:", ppl)
      print()

