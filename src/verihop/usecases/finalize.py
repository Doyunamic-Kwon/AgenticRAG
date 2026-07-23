"""[7]final_op + [8]Path-Check·체인중재 + [9]응답조립.
reads:  hops[], hop_results{hop_id:HopResult}, final_op
writes: Answer, path_check
may-import: verihop.models, verihop.ports (GraphPort), re (stdlib)

[9] 응답 조립은 결정적 템플릿으로 구현한다(LLM 1콜 대신). 이유: 조립 자체는 이미 확정된
hop_results를 문자열로 엮는 것뿐이라 LLM이 필요한 창의성이 없고, 템플릿이 더 테스트 가능하고
견고하다(ponytail — 필요해지면 표현 다듬기용 LLM 폴리시 패스를 여기 추가).
체인 중재(복수 체인 승자 결정, ADR-8)는 top-1만 쓰는 현재 범위에서는 해당 없음(단일 체인).
test: tests/usecase/test_finalize.py (fakes 주입)
done-check: tools/check_layers.sh 통과
"""
import re

_REF = re.compile(r"^\{(\d+)\}$")


def _refs(hop):
    return {int(m) for m in _REF.findall(hop["start"])} if _REF.match(hop["start"]) else set()


def _canon(name, alias):
    return alias.get(name, name)


def _path_check(hops, hop_results, graph, alias):
    """제약 BFS: hop 체인의 각 edge가 그래프에 실존하는지(RoG Algorithm 1 방식). 정보부족→None(중립)."""
    by_id = {h["id"]: h for h in hops}
    ordered = sorted(hops, key=lambda h: h["id"])
    cur_node = None
    for h in ordered:
        start = h.get("start_node") if not _refs(h) else cur_node
        r = hop_results.get(h["id"])
        if r is None or r["status"] == "FAILED" or not r["answer"]:
            return None
        ans_node = _canon(r["answer"], alias)
        if not (start and graph.exists(start) and graph.exists(ans_node)):
            return None
        edge = (graph.has_edge(start, ans_node, h["relation"]) if h["direction"] == "fwd"
                else graph.has_edge(ans_node, start, h["relation"]))
        if not edge:
            return False
        cur_node = ans_node
    return True


def _confidence(hops, hop_results, path_check_ok):
    ratios = []
    for h in hops:
        r = hop_results.get(h["id"])
        if not r or not r["candidates"]:
            continue
        top = max(r["candidates"], key=lambda c: c["score"])
        v = top["verification"]
        checks = [x for x in (v["type_check"], v["name_check"], v["desc_check"]) if x is not None]
        if v["backlink_check"] in ("pass", "contradict"):
            checks.append(v["backlink_check"] == "pass")
        if checks:
            ratios.append(sum(1 for c in checks if c) / len(checks))
    base = sum(ratios) / len(ratios) if ratios else 0.0
    return min(1.0, base + (0.1 if path_check_ok else 0.0))


def finalize(hops, hop_results, final_op, *, graph, alias):
    """returns Answer dict."""
    referenced = set().union(*[_refs(h) for h in hops]) if hops else set()
    leaves = [h for h in hops if h["id"] not in referenced]

    failed = [h for h in hops if hop_results.get(h["id"], {}).get("status") == "FAILED"]
    evidence = []
    for h in sorted(hops, key=lambda x: x["id"]):
        r = hop_results.get(h["id"])
        if r and r["candidates"]:
            top = max(r["candidates"], key=lambda c: c["score"])
            for pid in top["evidence_paragraph_ids"]:
                evidence.append({"paragraph_id": pid, "hop_id": h["id"]})

    if not failed and leaves:
        # final_op 결정적 비교 (해당 시). CHAIN이고 final_op 없으면 leaf 답을 그대로 채택.
        if final_op:
            text = _apply_final_op(final_op, hop_results)
            status = "ANSWERED" if text is not None else "PARTIAL"
            text = text if text is not None else "; ".join(
                f"hop{h['id']}={hop_results[h['id']]['answer']}" for h in leaves)
        else:
            text = hop_results[leaves[0]["id"]]["answer"]
            status = "ANSWERED"
        path_ok = _path_check(hops, hop_results, graph, alias)
        return {"text": text, "status": status, "confidence": _confidence(hops, hop_results, path_ok),
                "path_check": path_ok, "evidence": evidence, "verified_until": None}

    # 일부/전부 실패 → UNVERIFIABLE, verified_until = 실패 이전 마지막 hop
    ok_ids = sorted(h["id"] for h in hops if hop_results.get(h["id"], {}).get("status") == "OK")
    verified_until = ok_ids[-1] if ok_ids else None
    partial = "; ".join(f"hop{i}: {hop_results[i]['answer']}" for i in ok_ids) or "없음"
    fail_ids = sorted(h["id"] for h in failed)
    text = f"hop{verified_until}까지 확인: {partial}. 부족: hop{fail_ids} 실패"
    return {"text": text, "status": "UNVERIFIABLE", "confidence": _confidence(hops, hop_results, None),
            "path_check": None, "evidence": evidence, "verified_until": verified_until}


def _apply_final_op(final_op, hop_results):
    vals = [hop_results[i]["answer"] for i in final_op["in_"] if i in hop_results]
    if len(vals) != 2:
        return None
    op = final_op["operator"]
    try:
        if op in ("GREATER", "SMALLER"):
            a, b = float(vals[0].replace(",", "")), float(vals[1].replace(",", ""))
            return vals[0] if (a > b) == (op == "GREATER") else vals[1]
        if op == "SAME":
            return "yes" if vals[0] == vals[1] else "no"
        # EARLIER/LATER: 날짜 파싱은 W5+ (2Wiki 평가셋엔 미사용, MVP 스코프 밖)
        return None
    except Exception:
        return None


def demo():
    class FakeGraph:
        types = {"베토벤": "PERSON", "본": "LOCATION", "독일": "LOCATION"}
        edges = {("베토벤", "bornIn", "fwd"): [("본", [])], ("본", "locatedIn", "fwd"): [("독일", [])]}
        def exists(self, n): return n in self.types
        def has_edge(self, s, d, r):
            return any(n == d for n, _ in self.edges.get((s, r, "fwd"), []))

    hops = [{"id": 1, "start": "베토벤", "start_node": "베토벤", "relation": "bornIn", "direction": "fwd"},
            {"id": 2, "start": "{1}", "start_node": None, "relation": "locatedIn", "direction": "fwd"}]
    hop_results = {
        1: {"hop_id": 1, "answer": "본", "status": "OK",
            "candidates": [{"score": 3.0, "evidence_paragraph_ids": ["p1"],
                           "verification": {"type_check": True, "name_check": True, "desc_check": True,
                                            "backlink_check": "pass"}}]},
        2: {"hop_id": 2, "answer": "독일", "status": "OK",
            "candidates": [{"score": 3.0, "evidence_paragraph_ids": ["p2"],
                           "verification": {"type_check": True, "name_check": True, "desc_check": True,
                                            "backlink_check": "pass"}}]},
    }
    ans = finalize(hops, hop_results, None, graph=FakeGraph(), alias={})
    assert ans["text"] == "독일" and ans["status"] == "ANSWERED" and ans["path_check"] is True
    assert ans["confidence"] > 0.9, ans

    # 실패 케이스
    hop_results2 = {1: hop_results[1], 2: {**hop_results[2], "status": "FAILED"}}
    ans2 = finalize(hops, hop_results2, None, graph=FakeGraph(), alias={})
    assert ans2["status"] == "UNVERIFIABLE" and ans2["verified_until"] == 1
    print("finalize demo OK")


if __name__ == "__main__":
    demo()
