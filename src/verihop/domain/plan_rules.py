"""계획 검증 8종 — 순수 함수, I/O 없음 (03 [2]). mock 없이 단위테스트.
reads:  Hop[], FinalOp, goal_type_surface (스키마·op타입·alias는 caller가 인자로 주입)
writes: 검사별 pass/fail + 실패 처리 지시(재생성/SIMPLE/절단/final_op 제거) + execution_order
검사 8(C5): definition 내 '그 hop 답(tail)·후속 참조 엔티티'명 검출. 앵커·조건 서술의 기존 엔티티는 허용.
may-import: verihop.models, re (stdlib)
test: tests/unit/test_plan_rules.py
"""
import re
from verihop.models import Hop, FinalOp

_REF = re.compile(r"\{(\d+)\}")


def _refs(hop: Hop) -> set[int]:
    return {int(m) for m in _REF.findall(hop["start"])}


def _topo(hops: list[Hop]):
    """참조 위상정렬. 순환·미존재 참조면 None."""
    ids = {h["id"] for h in hops}
    by = {h["id"]: _refs(h) for h in hops}
    for h in hops:
        if not (_refs(h) <= ids):                     # dangling ref
            return None
    order, done = [], set()
    while len(done) < len(hops):
        ready = sorted(i for i in ids if i not in done and by[i] <= done)
        if not ready:                                 # cycle
            return None
        order += ready
        done.update(ready)
    return order


def _depth(hops: list[Hop]) -> int:
    by = {h["id"]: _refs(h) for h in hops}
    memo: dict[int, int] = {}

    def d(i, seen=frozenset()):
        if i in seen or i not in by:                  # 순환 방어
            return 0
        if i in memo:
            return memo[i]
        r = 1 + max((d(j, seen | {i}) for j in by[i]), default=0)
        memo[i] = r
        return r
    return max((d(h["id"]) for h in hops), default=0)


def validate_plan(hops, final_op, goal_type_surface, *,
                  schema_relations, op_types=None, alias_names=frozenset()):
    """returns {ok, failures:[(check#, hop_id, action)], execution_order}."""
    op_types = op_types or {}
    ids = {h["id"] for h in hops}
    F: list[tuple] = []

    for h in hops:                                    # 1 relation ∈ 스키마+relatedTo
        if h["relation"] not in schema_relations and h["relation"] != "relatedTo":
            F.append((1, h["id"], "재생성"))
        if h["direction"] not in ("fwd", "inv"):      # 2 direction
            F.append((2, h["id"], "재생성"))

    order = _topo(hops)                               # 3 참조 무결성 + DAG
    if order is None:
        F.append((3, None, "재생성"))
    if final_op:
        for i in final_op["in_"]:
            if i not in ids:
                F.append((3, None, "재생성"))

    if len(hops) > 5:                                 # 4 총hop≤5, 깊이≤3
        F.append((4, None, "절단/SIMPLE"))
    if _depth(hops) > 3:
        F.append((4, None, "절단/SIMPLE"))

    if goal_type_surface:                             # 5 타입체인: leaf(or op출력) == goal_type_surface
        referenced = set().union(*[_refs(h) for h in hops]) if hops else set()
        leaves = [h for h in hops if h["id"] not in referenced]
        if final_op is None and len(leaves) == 1:
            if leaves[0]["expected"]["type"] != goal_type_surface:
                F.append((5, leaves[0]["id"], "재생성"))

    if final_op:                                      # 6 op 입력 타입 == 연산자 요구 타입
        req = op_types.get(final_op["operator"])
        if req and req != "any":
            for i in final_op["in_"]:
                h = next((x for x in hops if x["id"] == i), None)
                if h and h["expected"]["type"] != req:
                    F.append((6, i, "final_op 제거"))

    for h in hops:                                    # 8 정답 누출 (C5): tail·후속참조 엔티티명이 definition에
        defn = h["expected"]["definition"]
        anchor = h["anchor_meta"]["name"] if h.get("anchor_meta") else None
        for name in alias_names:
            # 문제: 부분문자열 매칭이라 alias가 1~2음절짜리 짧은 이름이면("신","원","나") 흔한
            # 음절이 우연히 definition에 들어있다는 이유만으로 오탐(127문항 확대평가에서 검사8이
            # 92.9%→수정 후에도 상당수 계획을 계속 걸러내던 잔여 원인). 실제 답 고유명사는 대부분
            # 2음절 이상이라 1글자 alias는 애초에 제외. 앵커 비교도 완전일치 대신 포함관계로 완화
            # (LLM이 anchor_meta.name에 따옴표·접두어를 섞어 내보내는 포맷 흔들림 방어).
            if len(name) < 2:
                continue
            if anchor and (name in anchor or anchor in name):
                continue
            if name in defn:
                # 앵커가 아닌 alias 엔티티명이 정의에 있으면 누출 의심
                F.append((8, h["id"], "재생성"))
                break

    return {"ok": not F, "failures": F, "execution_order": order or []}


def _hop(i, start, rel="bornIn", d="fwd", typ="LOCATION", defn="조건", anchor=None):
    return {"id": i, "start": start, "relation": rel, "direction": d,
            "expected": {"type": typ, "definition": defn},
            "anchor_meta": {"name": anchor, "disambiguator": None}}


def demo():
    S = {"bornIn", "locatedIn", "createdBy"}
    # 시나리오 A: 베토벤-bornIn->도시, {1}-locatedIn->나라. leaf type=LOCATION==goal
    a = [_hop(1, "베토벤", "bornIn", typ="LOCATION", anchor="베토벤"),
         _hop(2, "{1}", "locatedIn", typ="LOCATION")]
    r = validate_plan(a, None, "LOCATION", schema_relations=S)
    assert r["ok"] and r["execution_order"] == [1, 2], r

    # 1 relation 위반
    bad = [_hop(1, "x", rel="unknownRel")]
    assert any(f[0] == 1 for f in validate_plan(bad, None, None, schema_relations=S)["failures"])

    # 3 순환 참조
    cyc = [_hop(1, "{2}"), _hop(2, "{1}")]
    assert any(f[0] == 3 for f in validate_plan(cyc, None, None, schema_relations=S)["failures"])

    # 3 미존재 참조
    dang = [_hop(1, "{9}")]
    assert any(f[0] == 3 for f in validate_plan(dang, None, None, schema_relations=S)["failures"])

    # 4 총hop>5
    many = [_hop(i, "x") for i in range(1, 7)]
    assert any(f[0] == 4 for f in validate_plan(many, None, None, schema_relations=S)["failures"])

    # 8 정답 누출: definition에 alias 엔티티('독일')가 있고 앵커 아님
    leak = [_hop(1, "베토벤", defn="독일에 있는 도시", anchor="베토벤")]
    assert any(f[0] == 8 for f in validate_plan(leak, None, None, schema_relations=S,
                                                alias_names={"독일", "베토벤"})["failures"])
    # 앵커명은 허용 (C5): '베토벤'만 있으면 누출 아님
    ok = [_hop(1, "베토벤", defn="베토벤이 태어난 도시", anchor="베토벤")]
    assert not any(f[0] == 8 for f in validate_plan(ok, None, None, schema_relations=S,
                                                    alias_names={"베토벤"})["failures"])
    print("plan_rules demo OK")


if __name__ == "__main__":
    demo()
