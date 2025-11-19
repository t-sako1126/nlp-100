import re
import requests

input = "jawiki-uk.txt"
output = "cp03/out/29.txt"

with open(input, 'r', encoding='utf-8') as f:
    text = f.read()

# 基礎情報テンプレートから国旗画像名を抽出
pattern = r'\{\{基礎情報.*?\n(.*?)\n\}\}'
match = re.search(pattern, text, re.DOTALL)

flag_file = None
if match:
    lines = match.group(1).split('\n') # 各行に分割
    for line in lines:
        m = re.match(r'\|\s*国旗画像\s*=\s*(.*)', line) # 国旗画像の行を抽出
        if m: 
            flag_file = m.group(1).strip() # 画像ファイル名を取得
            break

url = ""
if flag_file:
    endpoint = "https://www.mediawiki.org/w/api.php" # MediaWiki APIエンドポイント
    params = { # APIパラメータ
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": f"File:{flag_file}",
        "iiprop": "url"
    }
    headers = {"User-Agent": "100-knocks script (contact@example.com)"}
    r = requests.get(endpoint, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    try:
        data = r.json()
    except ValueError:
        print("Unexpected response:", r.text[:200])
        raise

pages = data["query"]["pages"] # ページ情報を取得
for page in pages.values():
    url = page["imageinfo"][0]["url"] # 画像URLを取得

with open(output, 'w', encoding='utf-8') as out:
    out.write(url + "\n")
