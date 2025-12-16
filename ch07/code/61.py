import pandas as pd
from collections import Counter
import tqdm

train = pd.read_csv("~/workspace/100_knocks/ch07/SST-2/train.tsv", sep="\t")
dev = pd.read_csv("~/workspace/100_knocks/ch07/SST-2/dev.tsv", sep="\t")

# 各文ごとに単語を数えて、指定の形式の dict を作る
for dataset in [train, dev]:
    for _, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset)):
        text = row.iloc[0]   # 1列目: 文
        label = row.iloc[1]  # 2列目: ラベル

        # 文を単語に分割して頻度を数える
        word_counter = Counter(text.split())

        sample = {
            "text": text,
            "label": str(label),
            "feature": dict(word_counter),
        }
        with open(f"ch07/out/61_{'train' if dataset is train else 'dev'}.txt", "a") as f:
            f.write(f"{sample}\n")
        


