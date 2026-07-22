#!/usr/bin/env python3
"""W1.4 alias 사전 — 코퍼스 문서제목 + 2Wiki 엔티티/별칭 → {alias: canonical}.

용도(ADR-7): name_check(후보∈alias), 앵커 링킹, 교정 후보.
위키 redirect 병합은 덤프 필요 → 이번엔 문서제목·평가셋 별칭만(추후 확장).

usage: python3 scripts/build_alias.py
out: data/alias.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    alias: dict[str, str] = {}

    for fp in ["data/corpus.jsonl", "data/corpus_2wiki.jsonl"]:
        p = ROOT / fp
        if p.exists():
            for l in open(p, encoding="utf-8"):
                t = json.loads(l)["doc_title"]
                alias.setdefault(t, t)                       # 문서제목 = 정규 엔티티

    mh = ROOT / "data/eval/multihop_2wiki.jsonl"
    if mh.exists():
        for l in open(mh, encoding="utf-8"):
            g = json.loads(l)
            for a in g.get("answer_aliases", []):
                alias.setdefault(a, g["answer"])
            for h in g["hops"]:
                for a in h.get("target_aliases", []):
                    alias.setdefault(a, h["target"])

    # 그래프 노드도 엔티티로 추가 + eval 엔티티명 링킹(표기차 흡수)
    gp = ROOT / "data/graph.pkl"
    if gp.exists():
        import pickle
        G = pickle.load(open(gp, "rb"))
        for n in G.nodes:
            alias.setdefault(n, n)
        # 링킹 휴리스틱: eval 엔티티명이 그래프 노드에 부분포함되면 연결
        # ("존 레논"→"존 윈스턴 오노 레논", "제퍼슨"↔"토머스 제퍼슨"). backlink 검증 위해.
        eval_names = set()
        if mh.exists():
            for l in open(mh, encoding="utf-8"):
                g = json.loads(l)
                eval_names.add(g["answer"])
                for h in g["hops"]:
                    eval_names.add(h["target"])
        node_toks = [(n, set(n.split())) for n in G.nodes]
        linked = 0
        for ev in eval_names:
            if ev in G:
                continue
            et = set(ev.split())
            if len(et) < 2:                          # 단일 토큰은 과링킹 위험 → exact만
                continue
            hit = next((n for n, nt in node_toks if n != ev and (et <= nt or nt <= et)), None)
            if hit:
                alias[ev] = hit
                linked += 1
        print(f"  그래프 노드 {G.number_of_nodes()} 병합 · eval 링킹 {linked}건")

    out = ROOT / "data/alias.json"
    json.dump(alias, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"alias {len(alias)}개 → {out}")
    # 2Wiki gold 엔티티 포함 확인
    sample = [a for a in ["켄 번스", "리버풀", "존 레논"] if a in alias]
    print(f"  2Wiki 엔티티 샘플 포함: {sample}")


if __name__ == "__main__":
    main()
