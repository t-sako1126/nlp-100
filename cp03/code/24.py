import re

input = "cp03/out/20.txt"
output = "cp03/out/24.txt"

with open(input, 'r', encoding='utf-8') as f:
    text = f.read()

pattern = r'\[\[(File|ファイル):([^|\]]+)' ## メディアファイル名を抽出する正規表現
files = re.findall(pattern, text)

with open(output, 'w', encoding='utf-8') as out:
    for _, filename in files:
        out.write(filename + "\n")
