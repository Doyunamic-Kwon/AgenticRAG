#!/usr/bin/env python3
"""W1.3 트리플 추출 → 지식그래프 (NetworkX). solar-pro 제약 추출.

문단에서 스키마 10종 관계만 (head, relation, tail)로 뽑아 DiGraph 구축.
- node: type(관계 head/tail에서 파생) · edge: relation, source_paragraph_id (backlink 검증 원천)
- 관계 방향 fwd = head→tail (예: createdBy는 WORK→PERSON, "영화는 감독이 만들었다").

usage:
  python3 scripts/build_graph.py --limit 500          # main 코퍼스 샘플 + 2wiki 전량
  python3 scripts/build_graph.py --verify 30           # 기존 그래프에서 30엣지 정확도 샘플(수검수 대체)
out: data/graph.pkl
"""
import sys, os, json, argparse, random, pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml
import networkx as nx
from dotenv import load_dotenv

KEY_ENV = {"upstage": "UPSTAGE_API_KEY", "openai": "OPENAI_API_KEY"}

SCHEMA_DESC = """- bornIn: (사람)→(지명). 예: (베토벤)→(본)
- diedIn: (사람)→(지명). 예: (모차르트)→(빈)
- locatedIn: (장소/지역)→(더 큰 지역). 예: (본)→(독일)
- capitalOf: (도시)→(나라/지역). 그 도시가 수도일 때만. 예: (서울)→(대한민국)
- nationality: (사람)→(나라). 예: (베토벤)→(독일)
- studiedAt: (사람)→(학교). 예: (오바마)→(하버드 대학교)
- memberOf: (사람)→(단체). 예: (메시)→(FC 바르셀로나)
- createdBy: (작품/저작물)→(만든 사람). 작품이 head다! 예: (교향곡 9번)→(베토벤), (미국 독립 선언서)→(제퍼슨)
- teacherOf: (사람)→(사람). 예: (하이든)→(베토벤)
- occupation: (사람)→(직업/분야). 예: (베토벤)→(작곡가)"""

EXTRACT_PROMPT = """아래 문단에서 다음 관계의 사실만 (head, relation, tail) 트리플로 추출하라.

관계 스키마 (head 타입 → tail 타입, 방향 엄수):
{schema}

규칙:
- **오직 문단에 문장으로 쓰여 있는 사실만. 너의 사전지식으로 채우지 마라.**
  각 트리플에 근거 문장(evidence)을 넣되, 문단의 한 문장을 '...' 축약·생략 없이 통째로 그대로 복사하라.
  복사할 문장이 없으면 그 트리플을 넣지 마라.
- **타입·방향이 스키마와 맞아야 한다. 안 맞으면 넣지 마라.**
  (예: capitalOf의 head는 도시. createdBy의 head는 작품이고 사람이 아니다.)
- head/tail은 고유명사 엔티티. 대명사·일반명사·연도 금지.

문단: {text}

JSON만: {{"triples": [{{"head": "...", "relation": "createdBy", "tail": "...", "evidence": "문단에서 그대로 복사한 문장"}}]}}"""

VERIFY_PROMPT = """문단이 다음 트리플을 실제로 뒷받침하는가?
트리플: ({h}) --{r}--> ({t})   (의미: {desc})
문단: {text}
JSON만: {{"supported": true/false}}"""

REL_KO = {"bornIn": "태어난 곳", "diedIn": "죽은 곳", "locatedIn": "상위 지역",
          "capitalOf": "수도", "nationality": "국적", "studiedAt": "출신 학교",
          "memberOf": "소속", "createdBy": "만든 사람", "teacherOf": "스승", "occupation": "직업"}


def make_llm():
    load_dotenv(ROOT / ".env")
    c = yaml.safe_load(open(ROOT / "configs/settings.yaml", encoding="utf-8"))
    lp = c["providers"]["llm"]; lc = c["llm"][lp]
    from verihop.adapters.openai_llm import OpenAILLM
    return OpenAILLM(os.environ[KEY_ENV[lp]], lc["base_url"], lc["model"],
                     c["llm"].get("temperature", 0.0), c["llm"].get("max_retries", 2))


def load_relations():
    r = yaml.safe_load(open(ROOT / "configs/relations.yaml", encoding="utf-8"))
    return {x["name"]: (x["head"], x["tail"]) for x in r["relations"]}


def load_paras(limit):
    main = [json.loads(l) for l in open(ROOT / "data/corpus.jsonl", encoding="utf-8")]
    random.seed(0); random.shuffle(main)
    paras = main[:limit]
    extra = ROOT / "data/corpus_2wiki.jsonl"
    if extra.exists():                                # 2wiki gold 문서는 전량(eval 관련)
        paras += [json.loads(l) for l in open(extra, encoding="utf-8")]
    return paras


def build(limit):
    llm = make_llm()
    rel = load_relations()
    schema = set(rel)
    paras = load_paras(limit)
    print(f"트리플 추출 대상 {len(paras)}문단 (main {limit} + 2wiki)")

    G = nx.DiGraph()
    edges = conflicts = dropped = 0
    for i, p in enumerate(paras):
        try:
            out = llm.complete(EXTRACT_PROMPT.format(schema=SCHEMA_DESC, text=p["text"][:2000]), schema=True)
        except Exception as e:
            print(f"  skip {p['paragraph_id']}: {e}"); continue
        norm = " ".join(p["text"].split())
        for t in (out.get("triples") or []) if isinstance(out, dict) else []:
            r, h, tl = t.get("relation"), (t.get("head") or "").strip(), (t.get("tail") or "").strip()
            if r not in schema or not h or not tl or h == tl:
                continue
            ev = " ".join((t.get("evidence") or "").split())
            if len(ev) < 8 or ev not in norm:         # 근거 문장이 문단에 없으면 폐기 (그라운딩 강제)
                dropped += 1; continue
            for node, typ in [(h, rel[r][0]), (tl, rel[r][1])]:
                if node not in G:
                    G.add_node(node, type=typ)
                elif G.nodes[node]["type"] != typ:
                    conflicts += 1                    # 타입 충돌(finding #10): 첫 타입 유지
            G.add_edge(h, tl, relation=r, source_paragraph_id=p["paragraph_id"], evidence=ev)
            edges += 1
        if i % 50 == 0:
            print(f"  {i}/{len(paras)} · 노드 {G.number_of_nodes()} 엣지 {edges}", flush=True)
        if i % 100 == 0 and i > 0:
            pickle.dump(G, open(ROOT / "data/graph.pkl", "wb"))    # 체크포인트(hang/kill 대비)

    pickle.dump(G, open(ROOT / "data/graph.pkl", "wb"))
    print(f"\n그래프 저장 → data/graph.pkl | 노드 {G.number_of_nodes()} 엣지 {G.number_of_edges()} "
          f"(타입충돌 {conflicts}, 근거없어 폐기 {dropped})")
    from collections import Counter
    rc = Counter(d["relation"] for _, _, d in G.edges(data=True))
    print("관계 분포:", dict(rc.most_common()))


def verify(n):
    """엣지 n개 샘플 → solar-pro가 source 문단으로 뒷받침 여부 판정 (수검수 대체, 정밀도 추정)."""
    llm = make_llm()
    G = pickle.load(open(ROOT / "data/graph.pkl", "rb"))
    pid2text = {json.loads(l)["paragraph_id"]: json.loads(l)["text"]
                for fp in ["data/corpus.jsonl", "data/corpus_2wiki.jsonl"]
                if (ROOT / fp).exists() for l in open(ROOT / fp, encoding="utf-8")}
    all_edges = list(G.edges(data=True))
    random.seed(1)
    sample = random.sample(all_edges, min(n, len(all_edges)))
    ok = 0
    for h, t, d in sample:
        txt = pid2text.get(d["source_paragraph_id"], "")
        out = llm.complete(VERIFY_PROMPT.format(h=h, r=d["relation"], t=t,
                                                desc=REL_KO.get(d["relation"], ""), text=txt[:2000]), schema=True)
        s = isinstance(out, dict) and out.get("supported")
        ok += bool(s)
        if not s:
            print(f"  ✗ ({h}) -{d['relation']}-> ({t})")
    print(f"\n정밀도(샘플 {len(sample)}): {ok}/{len(sample)} = {ok/len(sample):.0%} "
          f"({'PASS ≥85%' if ok/len(sample) >= 0.85 else '미달'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="main 코퍼스 샘플 문단 수")
    ap.add_argument("--verify", type=int, default=0, help="기존 그래프 엣지 정확도 샘플 수")
    a = ap.parse_args()
    verify(a.verify) if a.verify else build(a.limit)
