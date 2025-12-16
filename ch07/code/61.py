import os
import json

import pandas as pd
from collections import Counter
import tqdm

train = pd.read_csv("ch07/SST-2/train.tsv", sep="\t")
dev = pd.read_csv("ch07/SST-2/dev.tsv", sep="\t")

os.makedirs("ch07/out", exist_ok=True)

# 各文ごとに単語を数えて、指定の形式の dict を作る
for name, dataset in [("train", train), ("dev", dev)]:
    rows = []
    for _, row in tqdm.tqdm(dataset.iterrows(), total=len(dataset), desc=name):
        text = row.iloc[0]   # 1列目: 文
        label = row.iloc[1]  # 2列目: ラベル

        # 文を単語に分割して頻度を数える
        word_counter = Counter(text.split())

        rows.append(
            {
                "text": text,
                "label": str(label),
                # feature は単語頻度 dict を JSON 文字列として入れる
                "feature": json.dumps(dict(word_counter), ensure_ascii=False),
            }
        )

    out_path = f"ch07/out/61/61_{name}.tsv"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
        


