import gensim
from dotenv import load_dotenv
import os
import pandas as pd
from tqdm import tqdm
from io import StringIO

load_dotenv()
path = os.getenv("06PATH")

model = gensim.models.KeyedVectors.load_word2vec_format(
    path,
    binary=True
)

lines = []
use_block = False

with open('/home/sakoda/workspace/100_knocks/ch06/questions-words.txt') as f:
    for line in f:
        line = line.strip()
        if line.startswith(':'):
            # セクション切り替え
            use_block = (line == ': capital-common-countries')
            continue

        if not use_block:
            continue

        if line:  # 空行はスキップ
            lines.append(line)

# 取り出した行だけを疑似ファイルにして pandas で読む
text = '\n'.join(lines)
df = pd.read_csv(
    StringIO(text),
    sep=r"\s+",
    header=None,
    names=['A', 'B', 'C', 'D']
)

analogy_list = []

for i in tqdm(range(len(df))):
    a = df.loc[i, 'A']
    b = df.loc[i, 'B']
    c = df.loc[i, 'C']

    word, sim = model.most_similar(
        positive=[b, c],
        negative=[a],
        topn=1
    )[0] 

    analogy_list.append((a, b, c, df.loc[i, 'D'], word, sim))

result_df = pd.DataFrame(
    analogy_list,
    columns=['A', 'B', 'C', 'D', 'pred', 'similarity']
)

# D と pred が同じかどうか（正解かどうか）を判定
result_df['is_correct'] = result_df['D'] == result_df['pred']

# print(result_df.to_string())

# 必要なら正解率も出せます
accuracy = result_df['is_correct'].mean()
print(f"Accuracy: {accuracy:.4f}")
