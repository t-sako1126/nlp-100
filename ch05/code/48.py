#!/usr/bin/env python3
# filepath: /home/sakoda/workspace/100_knocks/ch05/code/48.py
import numpy as np
from google import genai

PROMPT = """
あなたは川柳専門家として
以下の10の川柳に対して面白さを評価してください
10段階の評価で、次のことを評価基準としてください

・ユーモアや皮肉、風刺が効いているか
・人間の心理や行動を鋭く捉えているか
・軽妙洒脱で爽快感があるか
各川柳に対して、評価点数のみを数字で答えてください
1.  大掃除 見て見ぬふりの 来年へ
2.  実家では 誰もが子供に 逆戻り
3.  おせち見て 飽きる体に 罪悪感
4.  お年玉 渡す立場も 悪くねぇ
5.  仕事始め 頭はまだね 夢の中
6.  新年の 抱負を語る 妻の圧力
7.  年末の 挨拶メール 返信渋る
8.  箱根駅伝 なぜか涙腺 ゆるむ午後
9.  正月太り サイズ戻せず 春を待つ
10. SNS 新年の挨拶 誰とする
出力は「1 8 2 9 ... 10 8」のように番号と点を交互に半角スペース区切りで返すこと。
"""

client = genai.Client()
N_RUNS = 5

def ask_once():
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": PROMPT}]}],
    )
    tokens = resp.text.strip().split()
    scores = [int(tokens[i]) for i in range(1, len(tokens), 2)]
    return scores

runs = [ask_once() for _ in range(N_RUNS)]
arr = np.array(runs)

print("各回:", runs)
print("分散:", arr.var(axis=0))