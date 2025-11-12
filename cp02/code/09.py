with open("cp02/text/popular-names.txt", "r", encoding="utf-8") as f:
        lines = [line.rstrip().split("\t") for line in f]

lines.sort(key=lambda x: int(x[2]), reverse=True)

with open("cp02/out/09/sorted_09.txt", "w", encoding="utf-8") as out:
    for row in lines:
        out.write("\t".join(row) + "\n")