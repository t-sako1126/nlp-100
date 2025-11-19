N = 10

with open("cp02/text/popular-names.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        print(line, end="") 