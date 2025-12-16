import pandas as pd
import joblib
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# 学習済みモデルとベクトライザの読み込み
model = joblib.load("ch07/out/62/62_model.joblib")
vectorizer = joblib.load("ch07/out/62/62_vectorizer.joblib")

dev_df = pd.read_csv("ch07/out/61/61_dev.tsv", sep="\t")
dev_df.columns = ["text", "label", "feature"]
# dev_df 全体ではなく、先頭1行だけを使う
row = dev_df.iloc[0]

x_dict = [json.loads(row["feature"])]
y_true = [int(row["label"])]
X_dev = vectorizer.transform(x_dict)
y_pred = model.predict(X_dev)

print(y_true, y_pred)