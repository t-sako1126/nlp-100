import gzip
import json
import re
from collections import Counter
from tqdm import tqdm

import spacy

INPUT = "ch03/jawiki-country.json.gz"
N = 20
MAX_BYTES = 49149

def remove_markup(text: str) -> str:
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[(https?://[^\s\]]+)\s([^\]]+)\]", r"\2", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    return text

def split_text_by_bytes(text: str, max_bytes=MAX_BYTES):
    """SudachiPy の上限 49KB を超えないように UTF-8 バイト単位で分割"""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]

    chunks = []
    for i in range(0, len(encoded), max_bytes):
        chunk = encoded[i:i + max_bytes]
        chunks.append(chunk.decode("utf-8", errors="ignore"))
    return chunks

def main():
    nlp = spacy.load("ja_ginza")

    texts = []

    # ① gzipを読み込み → クリーニング → バイト分割
    with gzip.open(INPUT, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = remove_markup(record.get("text", ""))
            if not text:
                continue

            # ★ Sudachi 対策のバイト分割
            chunks = split_text_by_bytes(text)
            texts.extend(chunks)

    counter = Counter()

    # ② nlp.pipe で高速並列処理
    docs = nlp.pipe(
        texts,
        batch_size=200,
        n_process=4,
        disable=["ner", "parser"]  # 高速化
    )

    # ③ Bコードと同じフィルタ + lemma カウント
    for doc in tqdm(docs, total=len(texts), desc="形態素解析中"):
        for token in doc:
            if token.is_space or token.is_punct or token.like_num:
                continue
            lemma = token.lemma_.strip()
            if lemma:
                counter[lemma] += 1

    # ④ 結果を表示
    for word, freq in counter.most_common(N):
        print(f"{word}\t{freq}")

if __name__ == "__main__":
    main()
