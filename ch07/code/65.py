import joblib
from collections import Counter

text = "the worst movie I 've ever seen"

# 学習済みモデルとベクトライザを読み込み
model = joblib.load("ch07/out/62/62_model.joblib")
vectorizer = joblib.load("ch07/out/62/62_vectorizer.joblib")

# 文を単語に分割して頻度を数える
word_counter = Counter(text.split())

x_dict = [dict(word_counter)]
X = vectorizer.transform(x_dict)
y_pred = model.predict(X)
print("pred:", y_pred)