"""[1] 분해 통합 콜 (LLM 1콜, structured output) + [2] 계획 검증 8종 디스패치
reads:  corrected(잠정), 미해결 토큰+후보, relation 스키마
writes: hops[], final_op, goal_type_surface, corrected(확정), corrections(L4)
프리필터(03 [1]): 관계 표지 0~1개 + 단문 + 미해결 토큰 없음 → LLM 콜 생략, mode=SIMPLE.
실패 처리(03 [1]/[2]): 파싱 실패·구조 불량·계획검증 8종 실패 → 실패사유를 프롬프트에 얹어 재생성 1회
→ 그래도 실패면 mode=SIMPLE(빈 hops)로 강등.
may-import: verihop.models, verihop.ports (LLMPort), verihop.domain.plan_rules
test: tests/usecase/test_decompose.py (fakes 주입) — 본 파일 하단 demo()는 오프라인 고정응답 self-check.
  실 API(Upstage) 검증은 usecase가 adapters를 알면 안 되므로(ADR-10, check_layers.sh) 이 파일 밖의
  스크립트에서 OpenAILLM을 주입해 수행한다(합성 리포트에 결과 첨부).
done-check: tools/check_layers.sh 통과
"""
from __future__ import annotations
from verihop.ports import LLMPort
from verihop.domain.plan_rules import validate_plan

_SIMPLE_LEN = 15  # ponytail: "단문" 판정용 러프 글자수 임계값. eval 이후 실측 보정.

# 어미 근방 키워드 → goal_type_surface. 검산 전용(load-bearing 아님, 03 [1]), 러프 휴리스틱.
_GOAL_TYPE_HINTS = [
    ("나라", "LOCATION"), ("국가", "LOCATION"), ("도시", "LOCATION"),
    ("지역", "LOCATION"), ("수도", "LOCATION"),
    ("사람", "PERSON"), ("누구", "PERSON"), ("인물", "PERSON"),
    ("언제", "DATE"), ("연도", "DATE"), ("날짜", "DATE"), ("몇 년", "DATE"),
    ("몇 개", "NUMBER"), ("얼마", "NUMBER"), ("숫자", "NUMBER"),
    ("학교", "ORG"), ("단체", "ORG"), ("회사", "ORG"), ("소속", "ORG"),
    ("작품", "WORK"), ("영화", "WORK"), ("노래", "WORK"),
]

_OPS = {"EARLIER", "LATER", "SAME", "GREATER", "SMALLER"}

_CHECK_DESC = {
    1: "relation이 스키마(+relatedTo) 밖이다",
    2: "direction이 fwd/inv가 아니다",
    3: "start의 {n} 참조가 존재하지 않거나 순환/DAG 위반이다",
    4: "hop 총수가 5 초과이거나 참조 깊이가 3 초과다",
    5: "마지막 hop(leaf)의 expected.type이 goal_type_surface와 다르다",
    6: "final_op 입력 hop의 expected.type이 연산자 요구 타입과 다르다",
    8: "expected.definition에 그 hop의 답이나 후속 hop이 찾을 정답 고유명사가 노출됐다(앵커명은 허용)",
}


def _infer_goal_type(question: str) -> str | None:
    tail = question[-12:]
    for kw, t in _GOAL_TYPE_HINTS:
        if kw in tail:
            return t
    return None


def _kw_hits(question: str, schema_relations: dict) -> int:
    return sum(1 for rel in schema_relations.values() for kw in rel.get("ko", []) if kw in question)


def _looks_simple(question: str, unresolved: list, schema_relations: dict) -> bool:
    if unresolved:                       # L4 후보가 있으면 분해 콜이 유일한 해소 경로(ADR-7) → 콜 강제
        return False
    return _kw_hits(question, schema_relations) <= 1 and len(question) <= _SIMPLE_LEN


def _fmt_unresolved(unresolved: list) -> str:
    if not unresolved:
        return "없음"
    lines = []
    for item in unresolved:
        if isinstance(item, dict):
            tok, cands = item.get("token"), item.get("candidates", [])
        else:
            tok, cands = item[0], item[1]
        lines.append(f'- "{tok}" 후보: {list(cands)}')
    return "\n".join(lines)


def _build_prompt(question: str, unresolved: list, schema_relations: dict, types: list,
                  fail_note: str = "") -> str:
    rel_lines = "\n".join(
        f'- {name}: head={r["head"]}, tail={r["tail"]}, 한국어 표지={r.get("ko", [])}'
        for name, r in schema_relations.items())
    retry_txt = f"\n\n[직전 시도 실패 사유 — 반드시 고쳐서 다시 생성하라]\n{fail_note}" if fail_note else ""
    return f"""너는 한국어 질문을 지식그래프 다중 hop 질의 계획으로 분해하는 시스템이다.

[관계 스키마] relation은 반드시 아래 이름 중 하나이거나 "relatedTo"만 쓴다(창작 금지).
{rel_lines}

[허용 타입] {types}

[오타 후보] 원문의 미해결 토큰이다. 후보 중 하나로 교정하거나 원문을 유지하되, corrected_question에는
반드시 최종 결정을 반영한다.
{_fmt_unresolved(unresolved)}

[규칙]
1. relation은 스키마 이름 또는 "relatedTo"만 쓴다.
2. hop.start: 최초 hop은 리터럴 앵커 엔티티명, 이후 hop이 앞 hop의 답을 참조하면 "{{n}}"(n=참조 hop id) 형식만 쓴다.
3. direction: "fwd"(head→tail, 스키마 방향 그대로) 또는 "inv"(tail→head, 역방향)만 쓴다.
4. expected.definition은 "그 hop이 찾는 조건의 재서술" 문장만 담는다. **정답이 될 고유명사를 절대 포함하지
   않는다.** (허용 예: "베토벤이 태어난 도시" — 앵커 '베토벤'은 조건 서술의 일부라 허용. 금지 예: 그 도시의
   실제 이름을 definition에 적는 것)
5. anchor_meta.name은 hop 시작 앵커의 표면형이다(참조 hop이라 아직 모르면 null). disambiguator는 동명이인
   구분용 짧은 문구(모르면 null).
6. sub_query는 그 hop을 검색하기 위한 자연어 하위질문 1개다.
7. final_op은 여러 hop의 답을 비교해야 답이 나오는 질문(더 이르다/늦다/같다/크다/작다)에서만 채우고,
   아니면 null로 둔다. operator는 EARLIER|LATER|SAME|GREATER|SMALLER 중 하나, in은 비교 대상 hop id 배열이다.
8. corrected_question은 최종 확정 질문 문자열이다.
9. **질문에 이미 고유명사로 등장하는 엔티티는 그 자체를 hop.start로 바로 쓴다. 그 엔티티가 "누구에 의해
   만들어졌는지/누가 그를 만들었는지" 같은 불필요한 선행 hop을 지어내지 마라.** 질문이 이미 인물을
   이름으로 지목했다면(예: "OO의 곡을 부른 가수 OO는" 처럼 그 가수 이름이 문장에 그대로 있다면), 그
   인물을 만든 사람을 찾는 hop은 의미가 없다 — 그 인물 자신에서 바로 다음 관계로 넘어가라. 질문 속에
   이름이 명시된 엔티티를 다시 "찾아야 할 대상"으로 취급하지 마라.

[예시 1 — fwd, 참조 체인]
질문: "베토벤이 태어난 도시가 속한 나라는?"
출력: {{"corrected_question": "베토벤이 태어난 도시가 속한 나라는?", "final_op": null, "hops": [
  {{"id": 1, "start": "베토벤", "relation": "bornIn", "direction": "fwd",
    "expected": {{"type": "LOCATION", "definition": "베토벤이 태어난 도시"}},
    "anchor_meta": {{"name": "베토벤", "disambiguator": null}}, "sub_query": "베토벤이 태어난 도시는?"}},
  {{"id": 2, "start": "{{1}}", "relation": "locatedIn", "direction": "fwd",
    "expected": {{"type": "LOCATION", "definition": "그 도시가 속한 나라"}},
    "anchor_meta": {{"name": null, "disambiguator": null}}, "sub_query": "그 도시가 속한 나라는?"}}
]}}

[예시 2 — inv, 단일 hop]
질문: "잘츠부르크에서 태어난 음악가는 누구인가?"
출력: {{"corrected_question": "잘츠부르크에서 태어난 음악가는 누구인가?", "final_op": null, "hops": [
  {{"id": 1, "start": "잘츠부르크", "relation": "bornIn", "direction": "inv",
    "expected": {{"type": "PERSON", "definition": "잘츠부르크에서 태어난 음악가"}},
    "anchor_meta": {{"name": "잘츠부르크", "disambiguator": null}},
    "sub_query": "잘츠부르크에서 태어난 음악가는 누구인가?"}}
]}}

[예시 3 — 규칙 9: 인물이 이미 이름으로 등장 → 그 인물에서 바로 시작 (틀린 예 아님, 올바른 처리)]
질문: "존 레논의 곡 '하느님'을 부른 가수는 어디에서 태어났나요?"
설명: "존 레논"이 이미 이름으로 나와 있고, "그 곡을 부른 가수"는 존 레논 자신을 가리킨다(재서술일 뿐,
새로운 미지의 대상이 아니다). "누가 존 레논을 만들었는가" 같은 hop을 만들면 안 된다 — 바로 bornIn으로.
출력: {{"corrected_question": "존 레논의 곡 '하느님'을 부른 가수는 어디에서 태어났나요?", "final_op": null,
 "hops": [
  {{"id": 1, "start": "존 레논", "relation": "bornIn", "direction": "fwd",
    "expected": {{"type": "LOCATION", "definition": "존 레논이 태어난 곳"}},
    "anchor_meta": {{"name": "존 레논", "disambiguator": null}},
    "sub_query": "존 레논은 어디에서 태어났나?"}}
]}}

[질문] "{question}"

JSON 객체만 출력한다(설명·코드펜스 금지). 최상위 키: corrected_question, hops, final_op, corrections.
corrections는 오타를 교정했을 때만 [{{"from": "원문토큰", "to": "교정값"}}] 형식으로 채우고, 없으면 [].{retry_txt}"""


def _call_llm(llm: LLMPort, prompt: str) -> dict | None:
    try:
        out = llm.complete(prompt, schema=True)
    except Exception:
        return None
    return out if isinstance(out, dict) else None


def _coerce_hop(raw, fallback_id: int):
    if not isinstance(raw, dict):
        return None
    try:
        start, relation, direction = raw["start"], raw["relation"], raw["direction"]
        exp = raw["expected"]
        etype, edef = exp["type"], exp["definition"]
        sub_query = raw["sub_query"]
    except (KeyError, TypeError):
        return None
    if direction not in ("fwd", "inv") or not str(edef).strip() or not str(etype).strip():
        return None
    try:
        hid = int(raw.get("id", fallback_id))
    except (TypeError, ValueError):
        hid = fallback_id
    am = raw.get("anchor_meta")
    am = am if isinstance(am, dict) else {}
    return {
        "id": hid, "start": str(start), "start_node": None,
        "relation": str(relation), "relation_grounded": None,
        "direction": direction,
        "expected": {"type": str(etype), "definition": str(edef)},
        "anchor_meta": {"name": am.get("name"), "disambiguator": am.get("disambiguator")},
        "sub_query": str(sub_query), "hint_relations": None,
    }


def _coerce_hops(raw_hops) -> tuple[list, list]:
    if not isinstance(raw_hops, list):
        return [], ["hops가 리스트가 아니다"]
    hops, bad = [], []
    for i, raw in enumerate(raw_hops, start=1):
        h = _coerce_hop(raw, i)
        if h is None:
            bad.append(raw.get("id", i) if isinstance(raw, dict) else i)
        else:
            hops.append(h)
    return hops, bad


def _coerce_final_op(raw):
    if not isinstance(raw, dict):
        return None
    op = raw.get("operator")
    if op not in _OPS:
        return None
    ins = raw.get("in", raw.get("in_"))
    if not isinstance(ins, list) or not ins:
        return None
    try:
        return {"operator": op, "in_": [int(x) for x in ins]}
    except (TypeError, ValueError):
        return None


def _fold_l4(original_q: str, corrected_q: str, unresolved: list) -> list:
    """LLM의 corrected_question이 실제로 unresolved 후보 중 하나를 채택했는지 우리가 직접 diff해서
    Correction을 만든다(LLM이 부른 자체 'corrections' 필드는 신뢰하지 않는다 — 구조 불신 원칙 동일 적용)."""
    corrections = []
    for item in unresolved:
        if isinstance(item, dict):
            tok, cands = item.get("token"), item.get("candidates", [])
        else:
            tok, cands = item[0], item[1]
        if tok and tok in original_q and tok not in corrected_q:
            for c in cands:
                if c and c in corrected_q:
                    corrections.append({"from_": tok, "to": c, "source": "L4"})
                    break
    return corrections


def _format_failures(failures: list) -> str:
    return "; ".join(f"검사{c} 실패(hop {hid}): {_CHECK_DESC.get(c, '')}" for c, hid, _ in failures)


def decompose(corrected_question: str, unresolved: list, llm: LLMPort, *,
             schema_relations: dict, types: list, alias_names: set = frozenset(),
             op_types: dict | None = None) -> dict:
    """03 [1]+[2]. schema_relations: {relation_name: {"head","tail","ko":[...]}}.
    unresolved: [(token, [candidates])] — L3에서 못 고친 오타 후보(ADR-7)."""
    goal_type_surface = _infer_goal_type(corrected_question)

    def _simple(q: str = corrected_question) -> dict:
        return {"hops": [], "final_op": None, "goal_type_surface": goal_type_surface,
                "corrected_question": q, "corrections": [], "mode": "SIMPLE"}

    if _looks_simple(corrected_question, unresolved, schema_relations):
        return _simple()

    fail_note = ""
    for _ in range(2):                                     # 최초 1회 + 재생성 1회 (03 [1]/[2])
        prompt = _build_prompt(corrected_question, unresolved, schema_relations, types, fail_note)
        parsed = _call_llm(llm, prompt)
        if parsed is None:
            fail_note = "직전 출력이 유효한 JSON 객체가 아니었다. JSON 객체만 출력하라(코드펜스·설명 금지)."
            continue

        hops, bad = _coerce_hops(parsed.get("hops"))
        if bad:
            fail_note = (f"hop 필드 누락/형식 오류(id={bad}). 모든 hop에 id,start,relation,direction,"
                        f"expected.type,expected.definition,sub_query를 채워라.")
            continue
        if not hops:                                        # LLM이 분해 불필요로 판단 → SIMPLE
            return _simple(str(parsed.get("corrected_question") or corrected_question))

        final_op = _coerce_final_op(parsed.get("final_op"))
        new_q = str(parsed.get("corrected_question") or corrected_question)
        corrections = _fold_l4(corrected_question, new_q, unresolved)

        check = validate_plan(hops, final_op, goal_type_surface, schema_relations=schema_relations,
                              op_types=op_types, alias_names=alias_names)
        if check["ok"]:
            return {"hops": hops, "final_op": final_op, "goal_type_surface": goal_type_surface,
                    "corrected_question": new_q, "corrections": corrections, "mode": "FULL"}
        fail_note = _format_failures(check["failures"])

    return _simple()                                        # 재생성 후에도 실패 → SIMPLE 강등


class _FakeLLM:
    """오프라인 데모 전용 고정응답 LLM(LLMPort 프로토콜 덕타이핑). 실 API 검증은 usecases 밖에서
    OpenAILLM(어댑터)을 주입해 수행한다 — usecase는 adapters의 존재를 모른다(ADR-10)."""
    def __init__(self, responses):
        self._responses = list(responses)

    def complete(self, prompt, *, schema=None):
        if not self._responses:
            raise AssertionError("fake LLM: 예상 밖의 콜 (프리필터가 새지 않았는지 확인)")
        return self._responses.pop(0)


def demo():
    S = {
        "bornIn": {"head": "PERSON", "tail": "LOCATION", "ko": ["태어난", "출생지", "출신"]},
        "locatedIn": {"head": "LOCATION", "tail": "LOCATION", "ko": ["속한", "위치한", "있는"]},
    }
    types = ["PERSON", "LOCATION", "ORG", "DATE", "NUMBER", "WORK", "TERM"]

    # 1) FULL: 정상 2-hop 응답
    fake = _FakeLLM([{
        "corrected_question": "베토벤이 태어난 도시가 속한 나라는?", "final_op": None,
        "hops": [
            {"id": 1, "start": "베토벤", "relation": "bornIn", "direction": "fwd",
             "expected": {"type": "LOCATION", "definition": "베토벤이 태어난 도시"},
             "anchor_meta": {"name": "베토벤", "disambiguator": None}, "sub_query": "베토벤이 태어난 도시는?"},
            {"id": 2, "start": "{1}", "relation": "locatedIn", "direction": "fwd",
             "expected": {"type": "LOCATION", "definition": "그 도시가 속한 나라"},
             "anchor_meta": {"name": None, "disambiguator": None}, "sub_query": "그 도시가 속한 나라는?"},
        ],
    }])
    out = decompose("베토벤이 태어난 도시가 속한 나라는?", [], fake,
                    schema_relations=S, types=types, alias_names={"베토벤"})
    assert out["mode"] == "FULL" and len(out["hops"]) == 2
    assert out["hops"][0]["relation"] == "bornIn" and out["hops"][1]["start"] == "{1}"
    assert out["goal_type_surface"] == "LOCATION"

    # 2) SIMPLE: 프리필터가 LLM 콜 자체를 막아야 함 (응답 0개 fake라 콜 나가면 AssertionError로 검출)
    out2 = decompose("대한민국의 수도는?", [], _FakeLLM([]), schema_relations=S, types=types)
    assert out2["mode"] == "SIMPLE" and out2["hops"] == []

    # 3) 재생성 1회 후에도 실패 → SIMPLE 강등
    bad_hop = {"id": 1, "start": "x", "relation": "unknownRel", "direction": "fwd",
              "expected": {"type": "LOCATION", "definition": "d"},
              "anchor_meta": {"name": "x", "disambiguator": None}, "sub_query": "q"}
    bad_fake = _FakeLLM([{"corrected_question": "x", "hops": [bad_hop]},
                         {"corrected_question": "x", "hops": [bad_hop]}])
    out3 = decompose("관계 표지가 태어난 그리고 속한 둘 다 있는 임의의 긴 질문 문장", [], bad_fake,
                     schema_relations=S, types=types)
    assert out3["mode"] == "SIMPLE"

    # 4) L4: 오타 후보 채택이 corrections로 기록되어야 함
    l4_fake = _FakeLLM([{
        "corrected_question": "베토벤이 태어난 도시는?", "final_op": None,
        "hops": [{"id": 1, "start": "베토벤", "relation": "bornIn", "direction": "fwd",
                  "expected": {"type": "LOCATION", "definition": "베토벤이 태어난 도시"},
                  "anchor_meta": {"name": "베토벤", "disambiguator": None},
                  "sub_query": "베토벤이 태어난 도시는?"}],
    }])
    out4 = decompose("배토벤이 태어난 도시는?", [("배토벤", ["베토벤"])], l4_fake,
                     schema_relations=S, types=types, alias_names={"베토벤"})
    assert out4["mode"] == "FULL"
    assert out4["corrections"] == [{"from_": "배토벤", "to": "베토벤", "source": "L4"}], out4["corrections"]

    print("decompose demo OK:", out["mode"], len(out["hops"]), "|", out2["mode"], "|", out3["mode"],
         "|", out4["mode"], out4["corrections"])


if __name__ == "__main__":
    demo()
