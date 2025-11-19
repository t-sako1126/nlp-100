with open("cp02/text/popular-names.txt", "r", encoding="utf-8") as f:
    line_count = sum(1 for line in f)

print(f"行数：{line_count}")
