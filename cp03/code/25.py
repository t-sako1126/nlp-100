import re

input = "cp03/out/20.txt"
output = "cp03/out/25.txt"

with open(input, 'r', encoding='utf-8') as f:
    text = f.read()

# テンプレート本体を抜き出す
pattern = r'\{\{基礎情報.*?\n(.*?)\n\}\}' # 基礎情報テンプレートの内容を抽出する正規表現
match = re.search(pattern, text, re.DOTALL) # re.DOTALLで改行を含む任意の文字にマッチ

info = {}
if match:
    content = match.group(1)
    lines = content.split('\n')
    for line in lines:
        m = re.match(r'\|\s*(.+?)\s*=\s*(.*)', line) # 各行からキーと値を抽出する正規表現
        if m:
            key, value = m.groups()
            info[key] = value

with open(output, 'w', encoding='utf-8') as out:
    for k, v in info.items():
        out.write(f"{k}\t{v}\n") 
