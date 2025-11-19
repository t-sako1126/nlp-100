from collections import Counter

with open("cp02/text/popular-names.txt", "r", encoding="utf-8") as f:
        col1 = [line.split("\t")[0] for line in f]

counter = Counter(col1)
for name, freq in counter.most_common():
        print(f"{freq}\t{name}")
