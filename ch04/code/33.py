import spacy
nlp = spacy.load("ja_ginza")

with open("ch04/melos.txt", 'r', encoding='utf-8') as f:
    text = f.read()

pairs = []
doc = nlp(text) # 形態素解析と構文解析の実行
for sent in doc.sents:
    for token in sent: 
        if token.head != token:  # 係り先が存在する場合
            print(f"{token.text}\t{token.head.text}") 