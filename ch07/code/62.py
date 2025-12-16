import json
import joblib

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


train_df = pd.read_csv("ch07/out/61/61_train.tsv", sep="\t")

x_dict = [json.loads(f) for f in train_df["feature"]]
y = train_df["label"].astype(int).tolist()

vectorizer = DictVectorizer()
X = vectorizer.fit_transform(x_dict)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 学習済みモデルとベクトライザを保存
joblib.dump(model, "ch07/out/62_model.joblib")
joblib.dump(vectorizer, "ch07/out/62_vectorizer.joblib")

