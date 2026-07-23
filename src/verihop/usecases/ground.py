"""[3] 그라운딩 (LLM 0~1콜) — anchor + relation을 그래프 실존 개체에 정렬 (ADR-6).
reads:  hops[], alias, graph relation 집합
writes: start_node, relation_grounded, hint_relations
may-import: verihop.models, verihop.ports (GraphPort, EmbedderPort, LLMPort), math, re (stdlib)
test: tests/usecase/test_ground.py (fakes 주입)
done-check: tools/check_layers.sh 통과
"""
import math
import re

_REF = re.compile(r"^\{(\d+)\}$")

TIEBREAK_PROMPT = """질문 의도상 다음 관계 이름 '{relation}'과 가장 가까운 것을 후보 중 하나만 골라라.
후보: {candidates}
JSON만: {{"choice": "후보 중 하나 그대로"}}"""


def _is_ref(start: str) -> bool:
    return bool(_REF.match(start))


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _ground_anchor(hop, graph, alias, embedder, anchor_sim_threshold):
    if _is_ref(hop["start"]):
        return None                                   # {n} 참조는 실행 시점에 해석 (ADR-3), 여기선 미해결
    name = hop["start"]
    anchor_name = (hop.get("anchor_meta") or {}).get("name") or name
    node = alias.get(name) or alias.get(anchor_name)
    if node and graph.exists(node):
        return node
    if graph.exists(name):
        return name
    nodes = graph.all_nodes()                         # alias exact 실패 → 임베딩 top-1 폴백
    if not nodes:
        return None
    vecs = embedder.embed([name] + nodes)
    qv, node_vs = vecs[0], vecs[1:]
    sims = [_cos(qv, nv) for nv in node_vs]
    best_i = max(range(len(sims)), key=lambda i: sims[i])
    return nodes[best_i] if sims[best_i] >= anchor_sim_threshold else None   # 미해결 → graph_miss


def _ground_relation(hop, start_node, graph, embedder, llm, relation_desc, theta_high, theta_low):
    relation = hop["relation"]
    candidates = graph.relations_of(start_node) if start_node else []
    if not candidates:
        return None, None                             # graph_miss/무이웃 → 힌트 없음(조회 실패 아님)
    if relation in candidates:
        return relation, None                          # exact
    texts = [relation_desc.get(relation, relation)] + [relation_desc.get(c, c) for c in candidates]
    vecs = embedder.embed(texts)
    qv, cand_vs = vecs[0], vecs[1:]
    sims = [_cos(qv, cv) for cv in cand_vs]
    best_i = max(range(len(sims)), key=lambda i: sims[i])
    best_sim, best_c = sims[best_i], candidates[best_i]
    if best_sim > theta_high:
        return best_c, None                            # 스냅
    if best_sim > theta_low:                           # 타이브레이크 (LLM 1콜, 객관식)
        try:
            out = llm.complete(TIEBREAK_PROMPT.format(relation=relation, candidates=candidates), schema=True)
            choice = out.get("choice") if isinstance(out, dict) else None
            if choice in candidates:
                return choice, None
        except Exception:
            pass
        return None, candidates                         # LLM 실패/무효 응답 → 포기 취급
    return None, candidates                             # 포기: relation_grounded=None + hint_relations


def ground(hops, *, graph, alias, embedder, llm, relation_desc=None,
           anchor_sim_threshold=0.70, theta_high=0.90, theta_low=0.75):
    """hops(list[Hop]) → 그라운딩된 사본 리스트. 원본은 불변."""
    relation_desc = relation_desc or {}
    out = []
    for hop in hops:
        h = dict(hop)
        start_node = _ground_anchor(h, graph, alias, embedder, anchor_sim_threshold)
        h["start_node"] = start_node
        relation_grounded, hint = _ground_relation(
            h, start_node, graph, embedder, llm, relation_desc, theta_high, theta_low)
        h["relation_grounded"] = relation_grounded
        h["hint_relations"] = hint
        out.append(h)
    return out


def demo():
    class FakeGraph:
        rel = {"베토벤": ["bornIn", "occupation"]}
        nodes = ["베토벤", "본", "독일"]
        def exists(self, n): return n in self.nodes
        def relations_of(self, n): return self.rel.get(n, [])
        def all_nodes(self): return self.nodes

    class FakeEmbedder:
        # 간단 벡터: 이름 길이·특정 문자 존재로 구분되는 저차원 고정 벡터(코사인 테스트용)
        table = {"베토벤": [1, 0, 0], "모차르트": [0.9, 0.1, 0], "미해결앵커": [0, 0, 1],
                 "bornIn": [1, 0, 0], "diedIn": [0.95, 0.05, 0], "occupation": [0, 1, 0],
                 "관계: bornIn": [1, 0, 0], "관계: diedIn": [0.95, 0.05, 0], "관계: occupation": [0, 1, 0]}
        def embed(self, texts, is_query=False):
            return [self.table.get(t, [0, 0, 0]) for t in texts]

    class FakeLLM:
        def complete(self, prompt, schema=None):
            return {"choice": "diedIn"}

    G, A, E, L = FakeGraph(), {"베토벤": "베토벤"}, FakeEmbedder(), FakeLLM()

    # 1. exact anchor + exact relation
    r = ground([{"id": 1, "start": "베토벤", "relation": "bornIn", "anchor_meta": {"name": "베토벤"}}],
              graph=G, alias=A, embedder=E, llm=L)
    assert r[0]["start_node"] == "베토벤" and r[0]["relation_grounded"] == "bornIn"

    # 2. {n} 참조는 start_node 미해결(실행시점으로 이연)
    r2 = ground([{"id": 2, "start": "{1}", "relation": "bornIn", "anchor_meta": {}}],
               graph=G, alias={}, embedder=E, llm=L)
    assert r2[0]["start_node"] is None

    # 3. relation 스냅 (diedIn 요청했지만 노드엔 diedIn 없음 → occupation류 무관, bornIn과 유사도 높은 것 스냅)
    r3 = ground([{"id": 3, "start": "베토벤", "relation": "diedIn", "anchor_meta": {"name": "베토벤"}}],
               graph=G, alias=A, embedder=E, llm=L,
               relation_desc={"bornIn": "관계: bornIn", "occupation": "관계: occupation"},
               theta_high=0.90)
    assert r3[0]["relation_grounded"] == "bornIn", r3    # cos(diedIn,bornIn)=0.95*1≈0.999 > 0.90 스냅

    # 4. 앵커 미존재 + 임베딩도 낮으면 graph_miss
    r4 = ground([{"id": 4, "start": "미해결앵커", "relation": "bornIn", "anchor_meta": {"name": "미해결앵커"}}],
               graph=G, alias={}, embedder=E, llm=L, anchor_sim_threshold=0.70)
    assert r4[0]["start_node"] is None and r4[0]["relation_grounded"] is None

    print("ground demo OK")


if __name__ == "__main__":
    demo()
