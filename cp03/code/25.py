import re

input = "cp03/out/20.txt"
output = "cp03/out/25.txt"

with open(input, 'r', encoding='utf-8') as f:
    text = f.read()

pattern = r'\{\{基礎情報.*?\n(.*?)\n\}\}' 
match = re.search(pattern, text, re.DOTALL) 

info = {}
if match:
    content = match.group(1)
    lines = content.split('\n')
    for line in lines:
        m = re.match(r'\|\s*(.+?)\s*=\s*(.*)', line) 
        if m:
            key, value = m.groups()
            info[key] = value

with open(output, 'w', encoding='utf-8') as out:
    for k, v in info.items():
        out.write(f"{k}\t{v}\n") 
