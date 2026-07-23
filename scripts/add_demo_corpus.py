#!/usr/bin/env python3
"""시나리오 A/B 데모 엔티티를 한국어 위키에서 fetch해 코퍼스 증분으로 추가 (M13 해소).

01_기획서 데모 시나리오가 요구하는 엔티티(모차르트·잘츠부르크·잘츠부르크 페스티벌·베토벤·본)가
KorQuAD 기반 코퍼스엔 없었다(문서 자체가 없음, 언급만 몇 번). 2wiki 적응 때 쓴 것과 같은 방식으로
온디맨드 fetch.

usage: python3 scripts/add_demo_corpus.py
out: data/corpus_demo.jsonl
"""
import json, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "VeriHop-research/0.1 (bootcamp; keonorg@gachon.ac.kr)"}

# 시나리오 A(베토벤/본/독일) + B(모차르트/잘츠부르크/잘츠부르크 페스티벌) 데모 엔티티
TITLES = ["볼프강 아마데우스 모차르트", "잘츠부르크", "잘츠부르크 페스티벌",
          "루트비히 판 베토벤", "본 (독일)"]


def ko_intro(title):
    u = "https://ko.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "extracts",
        "exintro": 1, "explaintext": 1, "redirects": 1, "format": "json"})
    d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=15))
    pg = list(d["query"]["pages"].values())[0]
    txt = (pg.get("extract") or "").strip()
    real_title = pg.get("title", title)
    return real_title, txt


def main():
    out = []
    for t in TITLES:
        real_title, txt = ko_intro(t)
        if not txt:
            print(f"  스킵(본문 없음): {t}")
            continue
        pid = f"demo::{real_title}"
        out.append({"paragraph_id": pid, "doc_title": real_title, "text": txt[:3000]})
        print(f"  {real_title}: {len(txt)}자")

    path = ROOT / "data/corpus_demo.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"완료: {len(out)}문서 → {path}")


if __name__ == "__main__":
    main()
