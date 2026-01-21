import pandas as pd
import torch
from p70 import token2id  

train_df = pd.read_csv("~/workspace/100_knocks/ch07/SST-2/train.tsv", sep="\t")
dev_df   = pd.read_csv("~/workspace/100_knocks/ch07/SST-2/dev.tsv", sep="\t")

# テキストをID列に変換
def text_to_ids(text, token2id): 
    tokens = text.split() 
    ids = [token2id[t] for t in tokens if t in token2id] 
    return torch.tensor(ids, dtype=torch.long)

def build_dataset(df, token2id):
    data = []

    for _, row in df.iterrows():
        text = row["sentence"]
        label = torch.tensor([float(row["label"])], dtype=torch.float32)

        input_ids = text_to_ids(text, token2id)
        if input_ids.numel() == 0:
            continue  

        data.append({
            "text": text,
            "label": label,
            "input_ids": input_ids
        })

    return data

train_data = build_dataset(train_df, token2id)
dev_data   = build_dataset(dev_df, token2id)

if __name__ == "__main__":
    print("train size:", len(train_data))
    print("dev size  :", len(dev_data))
    print(train_data[0])
