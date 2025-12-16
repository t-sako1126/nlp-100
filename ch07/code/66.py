import pandas as pd
import joblib
import json
from sklearn.metrics import confusion_matrix

# 学習済みモデルとベクトライザの読み込み
model = joblib.load("ch07/out/62/62_model.joblib")
vectorizer = joblib.load("ch07/out/62/62_vectorizer.joblib")

dev_df = pd.read_csv("ch07/out/61/61_dev.tsv", sep="\t")

y_true = dev_df["label"].astype(int).tolist()
x_dict = [json.loads(f) for f in dev_df["feature"]]
X_dev = vectorizer.transform(x_dict)
y_pred = model.predict(X_dev)

cm = confusion_matrix(y_true, y_pred)
print("confusion matrix:")
print(cm)