import pandas as pd
import gensim
from dotenv import load_dotenv
import os
from tqdm import tqdm
from scipy.stats import spearmanr

load_dotenv()
path = os.getenv("06PATH")

model = gensim.models.KeyedVectors.load_word2vec_format(
    path,
    binary=True
)

df = pd.read_csv("/home/sakoda/workspace/100_knocks/ch06/wordsim353/combined.csv")

pred_sims = []
gold_sims = []

for _, row in df.iterrows():
    w1, w2 = row["Word 1"], row["Word 2"]
    gold_sims.append(row["Human (mean)"])
    try:
        sim = model.similarity(w1, w2)
    except KeyError:
        sim = None
    
    pred_sims.append(sim)

valid_idx = [i for i, s in enumerate(pred_sims) if s is not None]
pred_valid = [pred_sims[i] for i in valid_idx]
gold_valid = [gold_sims[i] for i in valid_idx]

rho, pval = spearmanr(pred_valid, gold_valid)
print("Spearman correlation:", rho)
print("p-value:", pval)