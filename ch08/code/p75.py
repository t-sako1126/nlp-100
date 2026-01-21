import torch
import torch.nn as nn

vocab_size = 4
emb_dim = 5
embeddings = nn.Embedding(vocab_size, emb_dim)
# 0番目の単語なので，[0]をTensorに変換
word = torch.tensor([0])
embed_word = embeddings(word)
print(embed_word)
print(word.shape, '->', embed_word.shape)