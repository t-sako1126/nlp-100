from transformers import AutoTokenizer

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "The movie was full of incomprehensibilities."

print(tokenizer.tokenize(text))