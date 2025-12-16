import json

import joblib
import pandas as pd


# 学習済みモデルとベクトライザを読み込み
model = joblib.load("ch07/out/62/62_model.joblib")
vectorizer = joblib.load("ch07/out/62/62_vectorizer.joblib")

# 検証データ（dev）の先頭1件を読み込み
dev_df = pd.read_csv("ch07/out/61/61_dev.tsv", sep="\t")
row = dev_df.iloc[0]

# feature 列の JSON 文字列を dict に戻してベクトル化
x_dict = [json.loads(row["feature"])]
X = vectorizer.transform(x_dict)

# 各ラベル（0: ネガ, 1: ポジ）の条件付き確率を計算
proba = model.predict_proba(X)[0]

print("text:", row["text"])
print("true label:", row["label"])
print("P(y=0 | x):", proba[0])
print("P(y=1 | x):", proba[1])

