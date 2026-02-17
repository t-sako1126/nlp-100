from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

device="cuda"
MODEL="gpt2"
PROMPT="The movie was full of"

tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()

input=tok(PROMPT,return_tensors="pt").to(device)

def generate_text(**kwargs):
    generate_id=model.generate(**input,max_new_tokens=40,pad_token_id=tok.eos_token_id,**kwargs)
    print(tok.decode(generate_id[0],skip_special_tokens=True),"\n")

generate_text(do_sample=False)
generate_text(do_sample=True,temperature=0.5)
generate_text(do_sample=True,temperature=1.5)
