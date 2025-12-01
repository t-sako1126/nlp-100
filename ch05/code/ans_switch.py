import pandas as pd

path = "ch05/high_school_mathematics.csv"
df = pd.read_csv(path, header=None, names=["question", "A", "B", "C", "D", "answer"])

cols = ["A", "B", "C", "D"]
for idx, row in df.iterrows():
    correct = row["answer"].strip().upper()
    if correct not in cols:
        continue
    c_idx = cols.index(correct)
    if c_idx != 3:  # swap with D
        values = row[cols].tolist()
        values[c_idx], values[3] = values[3], values[c_idx]
        df.loc[idx, cols] = values
    df.loc[idx, "answer"] = "D"

df.to_csv(path, header=False, index=False)