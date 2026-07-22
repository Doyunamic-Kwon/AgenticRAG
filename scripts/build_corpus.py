#!/usr/bin/env python3
"""W1.1 코퍼스 구축 — corpus_doc_list의 KorQuAD 문서를 문단 단위로 data/corpus.jsonl에.

ponytail: KorQuAD context를 그대로 문단으로 쓴다(이미 적정 청크). 하이퍼링크 이웃은
위키 덤프가 필요해 이번엔 제외 — backlink 그래프는 KorQuAD 문서 범위로 한정된다(W1.3에서 반영).

usage: python3 scripts/build_corpus.py
out: data/corpus.jsonl  ({paragraph_id, doc_title, text})
"""
import json
from pathlib import Path
from collections import defaultdict

RAW = Path("data/raw")
DOC_LIST = Path("data/w0/corpus_doc_list.txt")
OUT = Path("data/corpus.jsonl")


def main():
    docs = {t for t in DOC_LIST.read_text(encoding="utf-8").splitlines() if t.strip()}
    print(f"대상 문서: {len(docs)}")

    idx = defaultdict(int)     # 문서별 문단 인덱스 (안정적 paragraph_id)
    seen = set()               # (title, text) 중복 제거
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for split in ["train", "dev"]:
            data = json.load(open(RAW / f"KorQuAD_v1.0_{split}.json", encoding="utf-8"))["data"]
            for doc in data:
                title = doc["title"]
                if title not in docs:
                    continue
                for para in doc["paragraphs"]:
                    text = para["context"].strip()
                    key = (title, text)
                    if key in seen:
                        continue
                    seen.add(key)
                    pid = f"{title}::p{idx[title]}"
                    idx[title] += 1
                    f.write(json.dumps(
                        {"paragraph_id": pid, "doc_title": title, "text": text},
                        ensure_ascii=False) + "\n")
                    n += 1
    print(f"corpus 문단: {n}  →  {OUT}")
    assert n > 0, "빈 코퍼스 — corpus_doc_list.txt 확인 (W0.1 먼저 실행)"


if __name__ == "__main__":
    main()
