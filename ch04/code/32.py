import MeCab

with open("ch04/melos.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    
tagger = MeCab.Tagger()
token = []
node = tagger.parseToNode(text)

while node:
    if node.surface:
        f = node.feature.split(',')
        token.append((node.surface, f[0])) # (表層形, 品詞)
    node = node.next 

result = []
for i in range(len(token) - 2):
    w1, p1 = token[i]
    w2, p2 = token[i + 1]
    w3, p3 = token[i + 2]
    if p1 == '名詞' and w2 == 'の' and p3 == '名詞':
        result.append(f"{w1}{w2}{w3}")

for r in result:
    print(r)

    