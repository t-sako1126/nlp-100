import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 学習データの読み込みとベクトル化
train_df = pd.read_csv("ch07/out/61/61_train.tsv", sep="\t")
x_dict = [json.loads(f) for f in train_df["feature"]]
y = train_df["label"].astype(int).tolist()
vectorizer = DictVectorizer()
X = vectorizer.fit_transform(x_dict)

# 検証データの読み込みとベクトル化
dev_df = pd.read_csv("ch07/out/61/61_dev.tsv", sep="\t")
x_dev_dict = [json.loads(f) for f in dev_df["feature"]]
X_dev = vectorizer.transform(x_dev_dict)
y_true = dev_df["label"].astype(int).tolist()

# 複数の正則化パラメータ C で正解率を評価
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
accuracies = []

for C in C_values:
	model = LogisticRegression(max_iter=1000, C=C)
	model.fit(X, y)
	y_dev_pred = model.predict(X_dev)
	acc = accuracy_score(y_true, y_dev_pred)
	accuracies.append(acc)
	print(f"C={C}: accuracy={acc:.4f}")

# 正則化パラメータを横軸、正解率を縦軸としたグラフを描画
plt.plot(C_values, accuracies, marker="o")
plt.xscale("log")
plt.xlabel("Regularization Parameter (C)")
plt.ylabel("Accuracy")
plt.title("Effect of Regularization on Accuracy")
plt.grid(True)
plt.tight_layout()

# 画像をファイルに出力
plt.savefig("ch07/out/69.png")