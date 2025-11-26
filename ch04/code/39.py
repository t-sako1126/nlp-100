#!/usr/bin/env python3
import gzip
import json
import re
from collections import Counter
import spacy
import matplotlib.pyplot as plt
import japanize_matplotlib  # ここで日本語フォントを登録
import numpy as np
from tqdm import tqdm

INPUT = "ch03/jawiki-country.json.gz"

# -----------------------------
#  マークアップ除去（問題36と同等）
# -----------------------------
def remove_markup(text: str) -> str:
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)       # [[...| 表記 ]]
    text = re.sub(r"\[(https?://[^\s\]]+)\s([^\]]+)\]", r"\2", text)    # [URL label]
    text = re.sub(r"<.*?>", "", text)                                   # <tag>
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)                          # {{テンプレート}}
    return text

# GiNZA は短文が速いのでチャンク化
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


# -----------------------------
#  Zipf プロット
# -----------------------------
def zipf_plot(counter):
    freqs = [f for _, f in counter.most_common()]
    ranks = np.arange(1, len(freqs) + 1)

    plt.figure(figsize=(8,6))
    plt.scatter(np.log(ranks), np.log(freqs), s=5)
    plt.xlabel("log(順位)")
    plt.ylabel("log(出現頻度)")
    plt.title("Zipfの法則（log-logプロット）")
    plt.tight_layout()
    plt.savefig("ch04/out/39.png", dpi=300)
    plt.close()


# -----------------------------
#  メイン処理
# -----------------------------
def main():
    # GiNZA を高速構成でロード（parser/ner を無効化）
    nlp = spacy.load("ja_ginza", disable=["parser", "ner", "textcat"])

    counter = Counter()

    with gzip.open(INPUT, "rt", encoding="utf-8") as f:
        for line in tqdm(f):
            record = json.loads(line)
            raw = remove_markup(record.get("text", ""))
            if not raw:
                continue

            # GiNZA は短文で速いのでチャンク処理
            for chunk in iter_chunks(raw):
                doc = nlp(chunk)
                for token in doc:
                    if token.is_space or token.is_punct or token.like_num:
                        continue
                    lemma = token.lemma_.strip()
                    if lemma:
                        counter[lemma] += 1

    # Zipf プロット
    zipf_plot(counter)


if __name__ == "__main__":
    main()
