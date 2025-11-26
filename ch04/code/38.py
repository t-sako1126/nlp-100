import gzip
import json
import re
import math
from collections import Counter
from tqdm import tqdm
import spacy

INPUT = "ch03/jawiki-country.json.gz"
TOPN = 20

def remove_markup(text: str) -> str:
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[(https?://[^\s\]]+)\s([^\]]+)\]", r"\2", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    return text

CHUNK_SIZE = 5000
def iter_chunks(text: str):
    buf = ""
    for line in text.splitlines(True):
        if len(buf.encode("utf-8")) + len(line.encode("utf-8")) > CHUNK_SIZE:
            if buf:
                yield buf
                buf = ""
        buf += line
    if buf:
        yield buf

def main():
    nlp = spacy.load("ja_ginza", disable=["parser", "ner", "textcat"])

    # 名詞の文書頻度
    doc_freq = Counter()

    # 「日本」記事の TF
    tf_japan = Counter()

    # 全記事数
    total_docs = 0

    with gzip.open(INPUT, "rt", encoding="utf-8") as f:
        for line in tqdm(f):
            record = json.loads(line)
            title = record.get("title", "")
            raw = remove_markup(record.get("text", ""))
            if not raw:
                continue

            total_docs += 1
            appeared = set()  # df用

            # 解析
            for chunk in iter_chunks(raw):
                doc = nlp(chunk)
                for token in doc:
                    if token.like_num or token.is_punct or token.is_space:
                        continue
                    if token.pos_ != "NOUN":
                        continue

                    lemma = token.lemma_.strip()
                    if not lemma:
                        continue

                    appeared.add(lemma)

                    # 「日本」記事だけ TFを取る
                    if title == "日本":
                        tf_japan[lemma] += 1

            # df更新
            for w in appeared:
                doc_freq[w] += 1

    # TF-IDF計算
    tfidf = {}
    for word, tf in tf_japan.items():
        df = doc_freq.get(word, 1)
        idf = math.log(total_docs / df)
        tfidf[word] = tf * idf

    # 上位20語表示
    print("\n=== 問題38：TF-IDF 上位20 ===")
    for w, score in sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:TOPN]:
        print(f"{w}\tTF={tf_japan[w]}\tDF={doc_freq[w]}\tTF-IDF={score:.3f}")

if __name__ == "__main__":
    main()
