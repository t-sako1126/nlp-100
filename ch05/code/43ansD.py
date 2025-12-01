import os
import pandas as pd
from google import genai

client = genai.Client()

JMMLU = "ch05/high_school_mathematics_ansD.csv"
MODEL = "gemini-2.5-flash"

df = pd.read_csv(JMMLU, header=None)
df.columns = ["question", "A", "B", "C", "D", "answer"]


problems_text = ""
for idx, row in df.iterrows():
    problems_text += f"""
[{idx+1}]
問題: {row['question']}
A: {row['A']}
B: {row['B']}
C: {row['C']}
D: {row['D']}
"""
prompt = f"""
以下に150問程度の選択問題があります。
各問題について、正しい選択肢の記号（A/B/C/D）を1文字だけ答えてください。

回答形式は必ず以下とすること：
番号,選択肢
1,A
2,C
3,D
...

では始めます。

{problems_text}

では上記の全ての問題について回答してください。
"""

response = client.models.generate_content(
    model=MODEL,
    contents=[{"role": "user", "parts": [{"text": prompt}]}],
)

result_text = response.candidates[0].content.parts[0].text

pred = {}
for line in result_text.splitlines():
    if "," in line:
        num, ans = line.split(",", 1)
        if num.isdigit():
            pred[int(num)-1] = ans.strip()[0]  # 1文字だけ

correct = 0
for i in range(len(df)):
    model_ans = pred.get(i, "?")
    true_ans = df.loc[i, "answer"]
    print(f"{i+1}: 予測={model_ans}, 正解={true_ans}")

    if model_ans == true_ans:
        correct += 1

accuracy = correct / len(df)
print(f"正解率: {accuracy:.2%}")