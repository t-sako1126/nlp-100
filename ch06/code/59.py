import os
from dotenv import load_dotenv

import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

load_dotenv()
path = os.getenv("06PATH")

model = gensim.models.KeyedVectors.load_word2vec_format(
    path,
    binary=True
)

df = pd.read_csv(
    "/home/sakoda/workspace/100_knocks/ch06/questions-words.txt",
    sep=r"\s+",
    header=None,
    names=['A', 'B', 'C', 'D'],
    comment=':'
)

# 国名を収集（D列だけを国名とみなす）
countries = set()
for word in df['D'].dropna():
    if isinstance(word, str):
        countries.add(word)

# 国名のベクトルを抽出（モデルに無い単語はスキップ）
country_vectors = {
    country: model[country]
    for country in countries
    if country in model.key_to_index
}

if not country_vectors:
    raise RuntimeError("モデルに存在する国名が1つも見つかりませんでした。D列の単語とモデルの語彙を確認してください。")

# ラベルとベクトル行列を作成（順序を固定するためソート）
labels = sorted(country_vectors.keys())
X = np.vstack([country_vectors[c] for c in labels])

# t-SNE による2次元可視化
tsne = TSNE(
    n_components=2,
    random_state=0,
    perplexity=30,
    init="random",
    learning_rate="auto"
)
X_2d = tsne.fit_transform(X)

# 可視化して保存
out_dir = os.path.join(os.path.dirname(__file__), "..", "out")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "59_tsne.png")

plt.figure(figsize=(12, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=10)

for (x, y, label) in zip(X_2d[:, 0], X_2d[:, 1], labels):
    plt.text(x, y, label, fontsize=8)

plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()

print(f"t-SNE plot saved to: {out_path}")