import re

INPUT = "cp03/out/20.txt"
OUTPUT = "cp03/out/21.txt"

with open(INPUT, 'r', encoding='utf-8') as f:
    text = f.read()

pattern = r'\[\[Category:[^\]]+\]\]'  # カテゴリを抽出する正規表現
categories = re.findall(pattern, text)

with open(OUTPUT, 'w', encoding='utf-8') as out_file:
    for category in categories:
        out_file.write(category + '\n')