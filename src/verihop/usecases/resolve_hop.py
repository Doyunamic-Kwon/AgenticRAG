"""resolve_hop = [4]검색+[5]검증확정+[6]재질의 캡슐화 = 원자 연산 (ADR-3/4).
reads:  Hop(그라운딩됨), state{vector,graph,llm,embedder,alias,tracer,pass_ratio,theta_desc,
        score_lambda,top_k,max_requery,tie_epsilon,relation_desc}
writes: HopResult (candidates 전량 보존, ADR-8)
may-import: verihop.models, verihop.ports, verihop.domain.verify_rules, math (stdlib)

backlink는 origin(vector/graph)과 무관하게 항상 균일 계산한다 — graph 출신이라고 자동 통과
처리하지 않는다(C1). graph 경로로 찾은 후보는 애초에 그 엣지로 발견됐으므로 재계산해도
자연히 "pass"가 나오고, 다른 후보가 진짜 답으로 확인되는 "contradict" 우회를 만들지 않는다.

캐시: [4] 검색결과만 대상(memo_cache, C6) — 이번 구현에는 아직 미배선(# ponytail: 성능 이슈
생기면 캐시 데코레이터를 [4] 검색 호출에만 씌운다. 검증·재질의는 항상 재실행).
verifier 주입: quad(domain.verify_rules.decide, 기본값) | llm_judge(Ours−G/Agent-basic arm, C4).
test: tests/usecase/test_resolve_hop.py (fakes 주입)
done-check: tools/check_layers.sh 통과
"""
import math
from verihop.domain import verify_rules as _quad

EXTRACT_PROMPT = """아래 문단들에서 다음 조건에 맞는 후보 엔티티를 추출하라.
타입: {type}
정의: {definition}

문단:
{paragraphs}

문단에 실제로 언급된 후보만. 없으면 빈 목록.
엔티티명에서 조사(은/는/이/가/을/를/의/에서/으로/와/과 등)는 반드시 떼고 원형만 적어라
(예: "베토벤은"이 아니라 "베토벤"). 조사가 남으면 alias 사전과 매칭이 안 돼 정상 후보가 탈락한다.
JSON만: {{"candidates": [{{"entity": "...", "paragraph_ids": ["p1"]}}]}}"""

REQUERY_PROMPT = """검색이 답을 찾지 못했다. 더 나은 검색 질의를 하나 제안하라.
원래 질문 조건: {definition}
실패 사유: {reasons}
이미 사용한 질의(반복 금지): {used}
그래프가 아는 관계 후보: {hints}
JSON만: {{"sub_query": "새 검색 질의"}}"""


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _canon(name, alias):
    return alias.get(name, name)


def _search(hop, sub_query, state):
    """[4] 경로A(벡터+LLM추출) + 경로B(그래프 이웃) → 병합 Candidate 목록."""
    vector, graph, llm, alias = state["vector"], state["graph"], state["llm"], state["alias"]
    top_k = state.get("top_k", 10)
    definition = hop["expected"]["definition"]
    query = f"{sub_query} {definition}" if state.get("use_hyde", True) else sub_query  # HyDE 결합

    path_a = vector.search(query, top_k)              # [(pid, text, score)]
    pid2text = {pid: text for pid, text, _ in path_a}  # desc_check가 ID가 아닌 실제 문단으로 비교하게
    paras_block = "\n".join(f"[{pid}] {text[:400]}" for pid, text, _ in path_a)
    a_candidates = []
    if path_a:
        try:
            out = llm.complete(EXTRACT_PROMPT.format(
                type=hop["expected"]["type"], definition=definition, paragraphs=paras_block), schema=True)
            score_by_pid = {pid: s for pid, _, s in path_a}
            for c in (out.get("candidates") or []) if isinstance(out, dict) else []:
                ent, pids = (c.get("entity") or "").strip(), c.get("paragraph_ids") or []
                if ent:
                    sc = max((score_by_pid.get(p, 0.0) for p in pids), default=0.0)
                    a_candidates.append({"entity": ent, "evidence_paragraph_ids": pids,
                                          "origin": "vector", "vector_score": sc})
        except Exception:
            pass

    path_b = []
    if hop.get("start_node") and hop.get("relation_grounded"):
        for node, pids in graph.neighbors(hop["start_node"], hop["relation_grounded"], hop["direction"]):
            path_b.append({"entity": node, "evidence_paragraph_ids": pids,
                            "origin": "graph", "vector_score": 0.0})
    full_text = state.get("pid2text", {})               # 그래프 근거 문단도 실제 텍스트로 (경로A 밖일 수 있음)
    for c in path_b:
        for pid in c["evidence_paragraph_ids"]:
            if pid not in pid2text and pid in full_text:
                pid2text[pid] = full_text[pid]

    merged: dict[str, dict] = {}
    for c in a_candidates + path_b:
        key = _canon(c["entity"], alias)
        if key not in merged:
            merged[key] = {"entity": c["entity"], "origin": c["origin"],
                            "evidence_paragraph_ids": list(c["evidence_paragraph_ids"]),
                            "vector_score": c["vector_score"]}
        else:
            m = merged[key]
            if m["origin"] != c["origin"]:
                m["origin"] = "both"
            for p in c["evidence_paragraph_ids"]:
                if p not in m["evidence_paragraph_ids"]:
                    m["evidence_paragraph_ids"].append(p)
            m["vector_score"] = max(m["vector_score"], c["vector_score"])
    return list(merged.values()), pid2text


def _backlink_state(hop, candidate_node, graph):
    """C1: 3값. 원본 relation(hop['relation'], grounded 아님) 기준으로 균일 계산."""
    start_node, relation, direction = hop.get("start_node"), hop["relation"], hop["direction"]
    if not start_node or not graph.exists(candidate_node):
        return None                                    # 시행 불가(약화모드)
    if direction == "fwd":
        edge = graph.has_edge(start_node, candidate_node, relation)
        real = [n for n, _ in graph.neighbors(start_node, relation, "fwd")]
    else:
        edge = graph.has_edge(candidate_node, start_node, relation)
        real = [n for n, _ in graph.neighbors(start_node, relation, "inv")]
    if edge:
        return "pass"
    if real and candidate_node not in real:
        return "contradict"                            # 그래프가 다른 진짜 답을 앎 → 하드 veto 대상
    return "absent"                                     # 그래프에 정보 없음 → 중립


def _verify(hop, candidates, pid2text, state, verifier):
    graph, alias = state["graph"], state["alias"]
    theta_desc = state.get("theta_desc", 0.60)
    pass_ratio = state.get("pass_ratio", 0.75)
    embedder = state["embedder"]
    expected_type = hop["expected"]["type"]              # falsy(""/None) = 타입 제약 없음(SIMPLE 폴백)
    defn_vec = embedder.embed([hop["expected"]["definition"]])[0]

    out = []
    for c in candidates:
        node = _canon(c["entity"], alias)
        node_type = graph.node_type(node)
        if not expected_type:
            type_ok = None                                # 제약 없음 → 시행 안 함(중립)
        else:
            type_ok = (node_type == expected_type) if node_type is not None else None
        name_ok = (c["entity"] in alias) or (node in alias.values()) or graph.exists(node)
        ev_text = " ".join(pid2text.get(p, "") for p in c["evidence_paragraph_ids"]).strip() or c["entity"]
        try:
            ev_vec = embedder.embed([ev_text])[0]
            desc_ok = _cos(ev_vec, defn_vec) > theta_desc
        except Exception:
            desc_ok = False
        backlink = _backlink_state(hop, node, graph)
        weak = not (hop.get("start_node") and graph.exists(node))

        verification = verifier(type_ok, name_ok, desc_ok, backlink, pass_ratio, weak)
        passed_count = sum(1 for v in (type_ok, name_ok, desc_ok) if v is True) + (1 if backlink == "pass" else 0)
        score = passed_count + state.get("score_lambda", 0.3) * c["vector_score"]
        out.append({"entity": c["entity"], "origin": c["origin"],
                    "evidence_paragraph_ids": c["evidence_paragraph_ids"],
                    "verification": verification, "score": score})
    return out


async def resolve_hop(hop, state, verifier=None):
    verifier = verifier or _quad.decide
    llm = state["llm"]
    max_requery = state.get("max_requery", 2)
    tie_epsilon = state.get("tie_epsilon", 0.15)

    sub_query = hop["sub_query"]
    queries_used, all_candidates, retries = [sub_query], [], 0

    while True:
        raw, pid2text = _search(hop, sub_query, state)
        scored = _verify(hop, raw, pid2text, state, verifier) if raw else []
        all_candidates.extend(scored)                   # 전량 보존 (ADR-8)
        scored.sort(key=lambda c: c["score"], reverse=True)

        if scored and scored[0]["verification"]["passed"]:
            tie = len(scored) > 1 and (scored[0]["score"] - scored[1]["score"]) < tie_epsilon
            return {"hop_id": hop["id"], "answer": scored[0]["entity"], "candidates": all_candidates,
                    "tie": tie, "branched": False, "retries": retries, "status": "OK",
                    "queries_used": queries_used}

        if retries >= max_requery:
            return {"hop_id": hop["id"], "answer": scored[0]["entity"] if scored else "",
                    "candidates": all_candidates, "tie": False, "branched": False,
                    "retries": retries, "status": "FAILED", "queries_used": queries_used}

        reasons = [c["verification"] for c in scored[:3]] or ["후보 없음"]
        try:
            out = llm.complete(REQUERY_PROMPT.format(
                definition=hop["expected"]["definition"], reasons=reasons, used=queries_used,
                hints=hop.get("hint_relations") or []), schema=True)
            new_q = (out.get("sub_query") or "").strip() if isinstance(out, dict) else ""
        except Exception:
            new_q = ""
        sub_query = new_q if (new_q and new_q not in queries_used) else sub_query + " 상세"
        queries_used.append(sub_query)
        retries += 1


def demo():
    import asyncio

    class FakeVector:
        def search(self, query, top_k):
            return [("p1", "베토벤은 본에서 태어났다.", 0.9), ("p2", "무관 문단.", 0.3)]

    class FakeGraph:
        edges = {("베토벤", "bornIn", "fwd"): [("본", ["p1"])]}
        types = {"본": "LOCATION", "베토벤": "PERSON"}
        def neighbors(self, node, rel, d):
            return self.edges.get((node, rel, d), [])
        def has_edge(self, s, d, r):
            return any(n == d for n, _ in self.edges.get((s, r, "fwd"), []))
        def node_type(self, n):
            return self.types.get(n)
        def exists(self, n):
            return n in self.types

    class FakeLLM:
        def complete(self, prompt, schema=None):
            if "후보 엔티티를 추출" in prompt:
                return {"candidates": [{"entity": "본", "paragraph_ids": ["p1"]}]}
            return {"sub_query": "베토벤 출생지 상세"}

    class FakeEmbedder:
        def embed(self, texts, is_query=False):
            return [[1.0, 0.0] if "본" in t or "태어" in t else [0.0, 1.0] for t in texts]

    hop = {"id": 1, "start": "베토벤", "start_node": "베토벤", "relation": "bornIn",
           "relation_grounded": "bornIn", "direction": "fwd",
           "expected": {"type": "LOCATION", "definition": "베토벤이 태어난 도시"},
           "sub_query": "베토벤 태어난 곳", "hint_relations": None}
    state = {"vector": FakeVector(), "graph": FakeGraph(), "llm": FakeLLM(), "embedder": FakeEmbedder(),
             "alias": {"본": "본"}, "pass_ratio": 0.75, "theta_desc": 0.5}

    r = asyncio.run(resolve_hop(hop, state))
    assert r["status"] == "OK" and r["answer"] == "본", r
    assert r["candidates"][0]["verification"]["backlink_check"] == "pass"

    # contradict 케이스: 그래프가 '본'을 진짜 답으로 아는데 후보가 '빈'이면 veto
    hop2 = {**hop, "start_node": "베토벤"}
    class FakeVector2:
        def search(self, q, k): return [("p3", "빈 관련 문단", 0.9)]
    class FakeLLM2:
        def complete(self, prompt, schema=None):
            if "추출" in prompt: return {"candidates": [{"entity": "빈", "paragraph_ids": ["p3"]}]}
            return {"sub_query": "재질의"}
    G2 = FakeGraph(); G2.types = {**G2.types, "빈": "LOCATION"}
    state2 = {**state, "vector": FakeVector2(), "llm": FakeLLM2(), "graph": G2, "max_requery": 0}
    r2 = asyncio.run(resolve_hop(hop2, state2))
    assert r2["candidates"][0]["verification"]["backlink_check"] == "contradict"
    assert r2["candidates"][0]["verification"]["passed"] is False       # 하드 veto 확인 (C1)
    print("resolve_hop demo OK")


if __name__ == "__main__":
    demo()
