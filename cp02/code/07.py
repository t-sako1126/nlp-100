with open ("cp02/text/popular-names.txt", "r", encoding="utf-8") as f:
    col1 = {line.split("\t")[0] for line in f}
    for name in sorted(col1):
        print(name)