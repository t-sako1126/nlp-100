import os
from dotenv import load_dotenv

import gensim
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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

# 国名を収集（D列だけを国名とみなす。suffix チェックはしない）
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

# k-means クラスタリング
k = 5  # クラスタ数
X = np.vstack(list(country_vectors.values()))  # 常に 2 次元にする
kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
kmeans.fit(X)

# 結果を整理して表示
result_df = pd.DataFrame({
    'country': list(country_vectors.keys()),
    'cluster': kmeans.labels_
}).sort_values(['cluster', 'country'])

print(result_df.to_string(index=False))
