from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
texts = [
"The movie was full of fun.",

"The movie was full of excitement.",

"The movie was full of crap.",

"The movie was full of rubbish.",
]

def get_cls_embedding(texts):
 inputs = tokenizer(texts, return_tensors="pt")
 outputs = model(**inputs)
 return outputs.last_hidden_state[:, 0, :]
 
print("コサイン類似度:\n")
for i in range(len(texts)):
 for j in range(i + 1, len(texts)):
  emb1 = get_cls_embedding([texts[i]])
  emb2 = get_cls_embedding([texts[j]])
  cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
  print(f"'{texts[i]}','{texts[j]}' = {cos_sim:.4f}\n")