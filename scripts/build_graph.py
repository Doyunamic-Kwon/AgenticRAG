#!/usr/bin/env python3
"""W1.3 트리플 추출 → 지식그래프 (NetworkX). solar-pro 제약 추출.

문단에서 스키마 10종 관계만 (head, relation, tail)로 뽑아 DiGraph 구축.
- node: type(관계 head/tail에서 파생) · edge: relation, source_paragraph_id (backlink 검증 원천)
- 관계 방향 fwd = head→tail (예: createdBy는 WORK→PERSON, "영화는 감독이 만들었다").

usage:
  python3 scripts/build_graph.py --limit 500          # main 코퍼스 샘플 + 2wiki 전량
  python3 scripts/build_graph.py --all --workers 12    # 전체 코퍼스, 동시 12개 (재개 가능)
  python3 scripts/build_graph.py --verify 30           # 기존 그래프에서 30엣지 정확도 샘플(수검수 대체)

재개: 기존 그래프의 G.graph['processed_pids']에 있는 문단은 스킵한다(엣지 0개로 끝난 문단도 포함) —
전체 코퍼스로 확장할 때 이미 처리한 552문단을 다시 돌리지 않는다.
out: data/graph.pkl
"""
import sys, os, json, argparse, random, pickle, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
- occupation: (사람)→(직업/분야). 예: (베토벤)→(작곡가)
- hostsEvent: (도시/장소)→(축제·행사). 그 도시에서 정기적으로 열리는 구체적 행사명일 때만. 예:
  (잘츠부르크)→(잘츠부르크 페스티벌)"""

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

VERIFY_PROMPT = """다음 트리플이 유효한지 세 가지 기준으로 판정하라.
트리플: ({h}) --{r}--> ({t})   (관계 의미: {desc})
문단: {text}

1. 그라운딩: 문단이 이 사실을 실제로 뒷받침하는가(추측·비약 아님).
2. 엔티티 형태: head와 tail이 실제 고유명사 엔티티인가? "~의 작품", "~의 데뷔곡", "~의 아버지" 같은
   서술구/별칭 문구는 무효다.
3. 관계 적합성: {r}의 의미에 head-tail 관계가 실제로 맞는가? (예: memberOf는 사람이 그 조직에 실제
   소속일 때만 — 장치·작품·사건이 장소·조직의 "멤버"인 경우는 없다. 애매하면 무효로 판정)

셋 다 만족해야 valid=true. 하나라도 걸리면 valid=false + 어느 기준에 걸렸는지 reason에 적어라.
JSON만: {{"valid": true/false, "reason": "grounding|entity_shape|relation_fit|ok"}}"""

REL_KO = {"bornIn": "태어난 곳", "diedIn": "죽은 곳", "locatedIn": "상위 지역",
          "capitalOf": "수도", "nationality": "국적", "studiedAt": "출신 학교",
          "memberOf": "소속", "createdBy": "만든 사람", "teacherOf": "스승", "occupation": "직업",
          "hostsEvent": "개최하는 축제/행사"}

# "그는 ~에서 태어났다"류 문장에서 실제 인물명 대신 대명사를 엔티티로 잘못 뽑는 사례가 실측 확인됨
# (evidence 그라운딩은 통과하지만 고유명사가 아니라 EXTRACT_PROMPT 규칙 위반 — 코드로 한 번 더 막는다).
PRONOUNS = {"그", "그녀", "그것", "이것", "저것", "여기", "거기", "저기", "이곳", "그곳", "저곳"}

# 코퍼런스 실패 시 실제 고유명사 대신 일반 범주어를 엔티티로 뽑는 사례(예: "그룹의 멤버"→"그룹",
# "영화에서는"→"영화")가 127문항 확대 평가에서 실측 확인됨 — decompose plan-check 8(정답 누출 방지)이
# alias 전체를 훑는데, 이런 범주어가 실제 엔티티처럼 alias에 섞이면 "그 감독이 태어난 곳" 같은 정상
# definition도 오탐 처리돼 SIMPLE로 강등(127개 중 92.9% 영향). occupation류 속성값(감독·가수 등)은
# 정당한 공유 값이라 여기서 안 막고 bootstrap._leak_check_alias_names에서 별도 제외.
GENERIC_TERMS = {"영화", "사람", "책", "소설", "그룹", "밴드"}


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


def load_paras(limit, all_paras):
    main = [json.loads(l) for l in open(ROOT / "data/corpus.jsonl", encoding="utf-8")]
    random.seed(0); random.shuffle(main)
    paras = main if all_paras else main[:limit]
    for extra_name in ["data/corpus_2wiki.jsonl", "data/corpus_demo.jsonl"]:
        extra = ROOT / extra_name
        if extra.exists():                            # 2wiki gold·데모 엔티티 문서는 전량
            paras += [json.loads(l) for l in open(extra, encoding="utf-8")]
    return paras


def _extract_one(llm, p, tries=3):
    """워커 스레드에서 실행. 성공 시 (p, triples, None), 실패 시 (p, None, 에러메시지).
    SDK 자체 재시도(max_retries)와 별개로 지속부하(sustained rate limit) 대비 수동 백오프."""
    last_err = None
    for i in range(tries):
        try:
            out = llm.complete(EXTRACT_PROMPT.format(schema=SCHEMA_DESC, text=p["text"][:2000]), schema=True)
            return p, ((out.get("triples") or []) if isinstance(out, dict) else []), None
        except Exception as e:
            last_err = str(e)[:200]
            if i < tries - 1:
                time.sleep(2 ** (i + 1))               # 2s, 4s
    return p, None, last_err


def build(limit, all_paras=False, workers=8, checkpoint_every=200):
    llm = make_llm()
    rel = load_relations()
    schema = set(rel)
    paras = load_paras(limit, all_paras)

    gpath = ROOT / "data/graph.pkl"
    if gpath.exists():                                # 재개: 이미 처리한 문단(엣지 0개 포함) 스킵
        G = pickle.load(open(gpath, "rb"))
        processed = G.graph.setdefault("processed_pids", set())
    else:
        G = nx.DiGraph()
        G.graph["processed_pids"] = processed = set()

    todo = [p for p in paras if p["paragraph_id"] not in processed]
    print(f"대상 {len(paras)}문단 · 이미 처리 {len(processed)} · 남은 작업 {len(todo)} · 동시 {workers}")
    if not todo:
        print("추가로 처리할 문단 없음."); pickle.dump(G, open(gpath, "wb")); return

    edges = conflicts = dropped = errors = 0
    err_samples = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_extract_one, llm, p): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            p, triples, err = fut.result()
            if triples is None:
                errors += 1                                # processed에 안 넣음 → 다음 실행에서 재시도
                if len(err_samples) < 5:
                    err_samples.append(err)
            else:
                processed.add(p["paragraph_id"])            # 성공한 문단만 "처리됨"으로 기록
                norm = " ".join(p["text"].split())
                for t in triples:
                    r, h, tl = t.get("relation"), (t.get("head") or "").strip(), (t.get("tail") or "").strip()
                    if (r not in schema or not h or not tl or h == tl
                            or h in PRONOUNS or tl in PRONOUNS
                            or h in GENERIC_TERMS or tl in GENERIC_TERMS):
                        continue
                    ev = " ".join((t.get("evidence") or "").split())
                    if len(ev) < 8 or ev not in norm:     # 근거 문장이 문단에 없으면 폐기(그라운딩 강제)
                        dropped += 1; continue
                    for node, typ in [(h, rel[r][0]), (tl, rel[r][1])]:
                        if node not in G:
                            G.add_node(node, type=typ)
                        elif G.nodes[node]["type"] != typ:
                            conflicts += 1                 # 타입 충돌(finding #10): 첫 타입 유지
                    G.add_edge(h, tl, relation=r, source_paragraph_id=p["paragraph_id"], evidence=ev)
                    edges += 1
            if i % 50 == 0:
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate if rate else 0
                print(f"  {i}/{len(todo)} · 노드 {G.number_of_nodes()} 엣지 {edges} · "
                      f"{rate:.1f}/s · ETA {eta/60:.0f}분", flush=True)
            if i % checkpoint_every == 0:
                pickle.dump(G, open(gpath, "wb"))          # 체크포인트(hang/kill 대비)

    pickle.dump(G, open(gpath, "wb"))
    print(f"\n그래프 저장 → data/graph.pkl | 노드 {G.number_of_nodes()} 엣지 {G.number_of_edges()} "
          f"(타입충돌 {conflicts}, 근거없어 폐기 {dropped}, 콜실패 {errors} — 미처리로 남아 다음 실행에 재시도) "
          f"· {time.time()-t0:.0f}s")
    if err_samples:
        print("실패 샘플:")
        for e in err_samples:
            print("  -", e)
    from collections import Counter
    rc = Counter(d["relation"] for _, _, d in G.edges(data=True))
    print("관계 분포:", dict(rc.most_common()))


def _load_pid2text():
    return {json.loads(l)["paragraph_id"]: json.loads(l)["text"]
            for fp in ["data/corpus.jsonl", "data/corpus_2wiki.jsonl", "data/corpus_demo.jsonl"]
            if (ROOT / fp).exists() for l in open(ROOT / fp, encoding="utf-8")}


def _judge_one(llm, h, t, d, pid2text, tries=3):
    """워커 스레드. returns (h, t, d, valid, reason)."""
    txt = pid2text.get(d["source_paragraph_id"], "")
    last_err = None
    for i in range(tries):
        try:
            out = llm.complete(VERIFY_PROMPT.format(h=h, r=d["relation"], t=t,
                               desc=REL_KO.get(d["relation"], ""), text=txt[:2000]), schema=True)
            valid = isinstance(out, dict) and bool(out.get("valid"))
            reason = out.get("reason", "?") if isinstance(out, dict) else "parse_error"
            return h, t, d, valid, reason
        except Exception as e:
            last_err = str(e)[:150]
            if i < tries - 1:
                time.sleep(2 ** (i + 1))
    return h, t, d, False, f"error:{last_err}"          # 판정 실패는 탈락 취급(보수적)


def verify(n, workers=6):
    """엣지 n개 샘플 → 3기준(그라운딩·엔티티형태·관계적합성) 재판정 (수검수 대체, 정밀도 추정)."""
    llm = make_llm()
    G = pickle.load(open(ROOT / "data/graph.pkl", "rb"))
    pid2text = _load_pid2text()
    all_edges = list(G.edges(data=True))
    random.seed(1)
    sample = random.sample(all_edges, min(n, len(all_edges)))

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_judge_one, llm, h, t, d, pid2text) for h, t, d in sample]
        for fut in as_completed(futures):
            h, t, d, valid, reason = fut.result()
            ok += valid
            if not valid:
                print(f"  ✗ [{reason}] ({h}) -{d['relation']}-> ({t})")
    print(f"\n정밀도(샘플 {len(sample)}): {ok}/{len(sample)} = {ok/len(sample):.0%} "
          f"({'PASS ≥85%' if ok/len(sample) >= 0.85 else '미달'})")


def prune(workers=6):
    """전체 엣지를 3기준으로 재판정해 탈락분을 제거한 새 그래프 저장 (Track 1).
    ADR-1 철학(생성-심사 분리)을 그래프 구축에도 적용 — 원본은 백업(되돌리기 가능)."""
    llm = make_llm()
    gpath = ROOT / "data/graph.pkl"
    G = pickle.load(open(gpath, "rb"))
    pid2text = _load_pid2text()
    all_edges = list(G.edges(data=True))
    print(f"전체 엣지 {len(all_edges)}개 재판정 (동시 {workers}) — 원본은 graph_preprune_backup.pkl로 백업")

    backup = ROOT / "data/graph_preprune_backup.pkl"
    pickle.dump(G, open(backup, "wb"))

    kept: list[tuple] = []
    reason_counts: dict[str, int] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_judge_one, llm, h, t, d, pid2text) for h, t, d in all_edges]
        for i, fut in enumerate(as_completed(futures), 1):
            h, t, d, valid, reason = fut.result()
            if valid:
                kept.append((h, t, d))
            else:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if i % 100 == 0:
                rate = i / (time.time() - t0)
                eta = (len(all_edges) - i) / rate if rate else 0
                print(f"  {i}/{len(all_edges)} · 유지 {len(kept)} · {rate:.1f}/s · ETA {eta/60:.0f}분", flush=True)

    G2 = nx.DiGraph()
    G2.graph["processed_pids"] = G.graph.get("processed_pids", set())
    for h, t, d in kept:
        for node in (h, t):
            if node not in G2:
                G2.add_node(node, type=G.nodes[node]["type"])
        G2.add_edge(h, t, **d)
    pickle.dump(G2, open(gpath, "wb"))

    print(f"\n가지치기 완료 · {time.time()-t0:.0f}s")
    print(f"엣지: {len(all_edges)} → {len(kept)} ({len(kept)/len(all_edges):.0%} 유지)")
    print(f"노드: {G.number_of_nodes()} → {G2.number_of_nodes()}")
    print("탈락 사유 분포:", dict(sorted(reason_counts.items(), key=lambda x: -x[1])))
    print(f"복구하려면: cp {backup} {gpath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="main 코퍼스 샘플 문단 수 (--all 이면 무시)")
    ap.add_argument("--all", action="store_true", help="main 코퍼스 전체 처리")
    ap.add_argument("--workers", type=int, default=8, help="동시 LLM 호출 수")
    ap.add_argument("--verify", type=int, default=0, help="기존 그래프 엣지 정확도 샘플 수")
    ap.add_argument("--prune", action="store_true", help="전체 엣지 재판정 후 탈락분 제거 (Track 1)")
    a = ap.parse_args()
    if a.prune:
        prune(workers=a.workers)
    elif a.verify:
        verify(a.verify, workers=a.workers)
    else:
        build(a.limit, all_paras=a.all, workers=a.workers)
