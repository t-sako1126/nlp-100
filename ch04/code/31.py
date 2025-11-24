import MeCab

with open("ch04/melos.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    
tagger = MeCab.Tagger()

verbs = []
verbs_base = []
node = tagger.parseToNode(text)
# 品詞，品詞細分類1，品詞細分類2，品詞細分類3，活用形，活用型，原形，読み，発音
while node:
    features = node.feature.split(',')
    if features[0] == '動詞':
        verbs.append(node.surface)
        verbs_base.append(features[6])
    node = node.next
    
for verb, base in zip(verbs, verbs_base):
    print(f"{verb}\t{base}")
    