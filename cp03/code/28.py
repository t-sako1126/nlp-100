import re

input = "cp03/out/20.txt"
output = "cp03/out/28.txt"

with open(input, 'r', encoding='utf-8') as f:
    text = f.read()

pattern = r'\{\{基礎情報.*?\n(.*?)\n\}\}'
match = re.search(pattern, text, re.DOTALL)

info = {}
if match:
    content = match.group(1)
    lines = content.split('\n')
    for line in lines:
        m = re.match(r'\|\s*(.+?)\s*=\s*(.*)', line) # 各行からキーと値を抽出する正規表現
        if m:
            key, value = m.groups()

            # 強調除去
            value = re.sub(r"'{2,5}", "", value) # 強調マークアップを削除する正規表現

            # 内部リンク
            value = re.sub(r'\[\[([^|\]]+\|)?([^|\]]+)\]\]', r'\2', value) # 内部リンクを表示名に置換する正規表現

            # <br> などのタグ除去
            value = re.sub(r'<.*?>', '', value) # HTMLタグを削除する正規表現

            info[key] = value

with open(output, 'w', encoding='utf-8') as out:
    for k, v in info.items():
        out.write(f"{k}\t{v}\n")
