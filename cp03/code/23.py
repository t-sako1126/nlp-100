import re

input = "cp03/out/20.txt"
output = "cp03/out/23.txt"

with open(input, 'r', encoding='utf-8') as f:
    text = f.read()
    
# == セクション名 == → level=1
pattern = r'(={2,})\s*(.+?)\s*\1'
sections = re.findall(pattern, text)

with open(output, 'w', encoding='utf-8') as out_file:
    for eq, name in sections:
        level = len(eq) - 1
        out_file.write(f'Level {level}: {name}\n')