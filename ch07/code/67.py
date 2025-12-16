import pandas as pd
import joblib
import json
from sklearn.metrics import classification_report

# 学習済みモデルとベクトライザの読み込み
model = joblib.load("ch07/out/62/62_model.joblib")
vectorizer = joblib.load("ch07/out/62/62_vectorizer.joblib")

train_df = pd.read_csv("ch07/out/61/61_train.tsv", sep="\t")
dev_df = pd.read_csv("ch07/out/61/61_dev.tsv", sep="\t")

# train データでの評価
y_train_true = train_df["label"].astype(int).tolist()
x_train_dict = [json.loads(f) for f in train_df["feature"]]
X_train = vectorizer.transform(x_train_dict)
y_train_pred = model.predict(X_train)
report_train = classification_report(y_train_true, y_train_pred, digits=4)
print("classification report_train:")
print(report_train)

# dev データでの評価
y_dev_true = dev_df["label"].astype(int).tolist()
x_dev_dict = [json.loads(f) for f in dev_df["feature"]]
X_dev = vectorizer.transform(x_dev_dict)
y_dev_pred = model.predict(X_dev)
report_dev = classification_report(y_dev_true, y_dev_pred, digits=4)
print("classification report_dev:")
print(report_dev)