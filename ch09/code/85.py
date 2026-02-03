from transformers import AutoTokenizer
import pandas as pd

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

train_df = pd.read_csv("ch07/SST-2/train.tsv", sep="\t", header=0)
dev_df = pd.read_csv("ch07/SST-2/dev.tsv", sep="\t", header=0)

# テキストと極性ラベルの読み込み
def tokenize(text):
 return tokenizer.tokenize(text)

train_df["tokens"] = train_df["sentence"].apply(tokenize)
dev_df["tokens"] = dev_df["sentence"].apply(tokenize)

print("Train example:")
print(train_df["tokens"].head(1)) 
print("\nDev example:")
print(dev_df["tokens"].head(1))