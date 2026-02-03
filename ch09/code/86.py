from transformers import AutoTokenizer
import pandas as pd

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

train_df = pd.read_csv("ch07/SST-2/train.tsv", sep="\t", header=0)

# テキストの冒頭4つを取得
sentences = train_df["sentence"].tolist()[:4]

# padding=True でバッチ内の最長シーケンスに合わせる
# padding='max_length' で指定した長さに固定する
inputs = tokenizer(sentences, padding=True, return_tensors="pt")

# inputs['input_ids'] がパディングされた数値列
# inputs['attention_mask'] がパディングマスク（0がパディング部分）
print(inputs['input_ids'])
print(inputs['attention_mask'])
