from transformers import pipeline, AutoTokenizer
import pandas as pd

model_id = "bert-base-uncased"
generator = pipeline("fill-mask", model=model_id) # マスクされた単語予測のパイプラインを作成

text = "The movie was full of [MASK]."

outputs = generator(text) # マスクされた単語を予測
print(pd.DataFrame(outputs[:1])) #最も確率の高い予測結果1件を表示