import random

with open("cp02/text/popular-names.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

with open("cp02/out/06/shuffled_06.txt", "w", encoding="utf-8") as out:
    out.writelines(lines)

print("cp02/out/06/shuffled_06.txt に保存しました．")

#$ shuf cp02/text/popular-names.txt -o cp02/out/06/shuffled_shuf.txt 