import numpy as np
import torch
from gensim.models import KeyedVectors

FILE_PATH = "/home/fujiwara/workspace/nlp100-2025/chapter06/data/GoogleNews-vectors-negative300.bin.gz"

wv = KeyedVectors.load_word2vec_format(FILE_PATH, binary=True)

emb_dim = wv.vector_size #次元数
vocab_size = len(wv.key_to_index) #語彙数

emb = np.zeros((vocab_size + 1, emb_dim), dtype=np.float32)


token2id = {"<PAD>": 0}
id2token = {0: "<PAD>"}

for i, word in enumerate(wv.key_to_index.keys(), start=1): #start=1で1番から入れる
    emb[i] = wv[word]
    token2id[word] = i
    id2token[i] = word

emb = torch.tensor(emb)  # torch.Tensorに変換


if __name__ == "__main__":
    print(emb.shape)         