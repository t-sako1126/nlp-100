import joblib

# 学習済みモデルとベクトライザを読み込み
model = joblib.load("ch07/out/62/62_model.joblib")
vectorizer = joblib.load("ch07/out/62/62_vectorizer.joblib")

weights = model.coef_[0]
feature_names = vectorizer.get_feature_names_out()

top20_pos_indices = weights.argsort()[-20:][::-1]
top20_neg_indices = weights.argsort()[:20]

print("Top 20 positive features:")
for idx in top20_pos_indices:
    print(f"{feature_names[idx]}: {weights[idx]}")
print("\nTop 20 negative features:")
for idx in top20_neg_indices:
    print(f"{feature_names[idx]}: {weights[idx]}")