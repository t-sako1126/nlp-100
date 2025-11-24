import spacy
nlp = spacy.load("ja_ginza")

with open("ch04/melos.txt", 'r', encoding='utf-8') as f:
    text = f.read()

predicates = []
doc = nlp(text) # 形態素解析と構文解析の実行
for sent in doc.sents:
    for token in sent: 
        if token.dep_ == "nsubj" and token.text == "メロス":
            predicates.append(token.head.text)
            
for p in predicates:
    print(p)