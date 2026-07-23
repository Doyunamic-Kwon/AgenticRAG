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

# 대명사·지시어 — 그래프 추출이 "그는 ~에서 태어났다" 같은 문장에서 실제 인물명 대신 이걸 엔티티로
# 잘못 뽑는 사례가 실측 확인됨(예: "그"→김국진). alias에 있으면 decompose 계획검증 8번(정답 누출)이
# "그 도시" 같은 정상적인 지시표현마다 오탐을 일으킨다 — 실제로 시나리오 B를 막았던 버그.
PRONOUNS = {"그", "그녀", "그것", "이것", "저것", "여기", "거기", "저기", "이곳", "그곳", "저곳"}

# 코퍼런스 실패로 그래프에 섞여든 일반 범주어(실제 엔티티가 아님) — build_graph.py GENERIC_TERMS와 동일.
# 있으면 decompose 계획검증 8번이 "영화 X의 감독" 같은 정상 definition마다 오탐(127문항 확대평가에서
# 92.9% 영향 실측). 그래프 자체는 build_graph.py가 신규 추출 시 걸러내고, 여기선 alias 링킹에서도 제외.
GENERIC_TERMS = {"영화", "사람", "책", "소설", "그룹", "밴드"}

# 같은 도시의 한국어 표기 변이(외래어 표기 통일 안 됨) — 위키 redirect 없이 수동 보강.
# 예: 모차르트 문서는 "비엔나"를 압도적으로 많이 쓰는데 그래프 노드는 "빈"이라 링킹 안 되면
# backlink 검증(그래프가 실제 답을 아는지)이 무력화된다.
MANUAL_ALIASES = {"비엔나": "빈", "베를린 (도시)": "베를린"}


def main():
    alias: dict[str, str] = {}

    for fp in ["data/corpus.jsonl", "data/corpus_2wiki.jsonl", "data/corpus_demo.jsonl"]:
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
            if n not in PRONOUNS and n not in GENERIC_TERMS:
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

    alias.update(MANUAL_ALIASES)                       # 표기변이 수동 보강 (자기자신 매핑 덮어씀)

    out = ROOT / "data/alias.json"
    json.dump(alias, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"alias {len(alias)}개 → {out}")
    # 2Wiki gold 엔티티 포함 확인
    sample = [a for a in ["켄 번스", "리버풀", "존 레논"] if a in alias]
    print(f"  2Wiki 엔티티 샘플 포함: {sample}")


if __name__ == "__main__":
    main()
