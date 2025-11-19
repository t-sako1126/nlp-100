import re

input = "cp03/out/20.txt"
output = "cp03/out/22.txt"

with open(input, 'r', encoding='utf-8') as f:
    text = f.read()
    
pattern = r'\[\[Category:([^\]|]+)(?:\|[^\]]*)?\]\]'
categories = re.findall(pattern, text)

with open(output, 'w', encoding='utf-8') as out_file:
    for category in categories:
        out_file.write(category + '\n')