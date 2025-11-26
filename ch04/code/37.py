import gzip
import json
import re
from collections import Counter
from tqdm import tqdm
import spacy

INPUT = "ch03/jawiki-country.json.gz"
N = 20

# ★ GiNZA の重い機能をオフ（速度2〜3倍）
DISABLE = ["ner", "parser", "textcat", "tagger"]  # taggerを消すとpos_が消えるので注意
# ※名詞が必要なら "tagger" は消さない（pos_ が使えなくなる）

def remove_markup(text: str) -> str:
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[(https?://[^\s\]]+)\s([^\]]+)\]", r"\2", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    return text

# ★ GiNZA は短文が圧倒的に速い → チャンクを小さく
CHUNK_SIZE = 5000  # bytes (元の1/9くらいに縮めると2〜3倍高速)

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
    # ★ parser, NER を無効化して高速化（taggerは残す）
    nlp = spacy.load("ja_ginza", disable=["parser", "ner", "textcat"])
    counter = Counter()
    counter_noun = Counter()

    with gzip.open(INPUT, "rt", encoding="utf-8") as f:
        for line in tqdm(f):
            record = json.loads(line)
            raw = remove_markup(record.get("text", ""))
            if not raw:
                continue

            # ★ 長文→短文にすることで解析速度が大幅アップ
            for chunk in iter_chunks(raw):
                doc = nlp(chunk)

                for token in doc:
                    # ★ 無駄な条件チェックを最小化
                    if token.like_num or token.is_punct or token.is_space:
                        continue

                    lemma = token.lemma_.strip()
                    if not lemma:
                        continue

                    counter[lemma] += 1

                    # 問題37：名詞頻度
                    if token.pos_ == "NOUN":
                        counter_noun[lemma] += 1

    print("\n=== 問題37: 名詞頻度 Top20 ===")
    for word, freq in counter_noun.most_common(N):
        print(word, freq)

if __name__ == "__main__":
    main()
