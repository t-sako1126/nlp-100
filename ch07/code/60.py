import pandas as pd

train = pd.read_csv("~/workspace/100_knocks/ch07/SST-2/train.tsv", sep="\t")
dev = pd.read_csv("~/workspace/100_knocks/ch07/SST-2/dev.tsv", sep="\t")

print(train.iloc[:, 1].value_counts())
print(dev.iloc[:, 1].value_counts())
