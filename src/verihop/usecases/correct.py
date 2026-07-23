"""[0] 교정 L1~L3 (결정적, LLM 금지)
reads:  raw_question, alias 사전 (dict[표제어→정규형], 위키 redirect 겸용)
writes: corrected_question(잠정), corrections, 미해결 토큰+후보
may-import: verihop.models (미사용 — 이 모듈은 stdlib만 씀: unicodedata, re. 편집거리는 직접 구현, ADR-7)
test: demo()/__main__ (이 프로젝트의 경량 테스트 관례, plan_rules.py/verify_rules.py와 동일)
done-check: tools/check_layers.sh 통과
"""
# W2.3. usecase는 함수(클래스 아님). 이 모듈은 ports 없이 순수 함수 — L1~L3는 결정적 문자열
# 처리뿐이라 LLM·임베더 호출이 없다(ADR-7). alias는 그래프 실존 엔티티 사전이라 그 자체가
# 인자로 충분하고 EmbedderPort 주입은 불필요.
import re
import unicodedata

# 조사 strip 후보. 길이 내림차순으로 검사해야 '에서'가 '에'보다 먼저 매치된다.
_PARTICLES = sorted(
    ["이", "가", "은", "는", "을", "를", "의", "에서", "으로", "로", "에"],
    key=len, reverse=True,
)


def _strip_particle(token: str) -> str:
    """흔한 조사 어미 1개 제거. 2자 미만으로 줄어들면 원형 유지 (스펙)."""
    for p in _PARTICLES:
        if token.endswith(p) and len(token) - len(p) >= 2:
            return token[: -len(p)]
    return token


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 거리. dict 조회 규모(alias ~수천 개)라 외부 라이브러리 없이 직접 구현."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _in_alias(stem: str, alias: dict[str, str]) -> bool:
    """L2 판정: stem이 alias 키 그 자체이거나, 어떤 alias 키의 부분문자열/접두 매치인지."""
    if not stem:
        return False
    if stem in alias:
        return True
    return any(stem in key or key.startswith(stem) for key in alias)


def correct(raw_question: str, alias: dict[str, str]) -> dict:
    # L1: NFC 정규화 + 공백 정리.
    # ponytail: no abbrev dict yet, add when needed
    q = unicodedata.normalize("NFC", raw_question)
    q = re.sub(r"\s+", " ", q).strip()

    tokens = q.split(" ") if q else []
    stems = [_strip_particle(t) for t in tokens]

    # L2: 전 토큰이 alias에 있거나 alias 키의 부분/접두 매치면 오타 없음 → L3 스킵.
    if tokens and all(_in_alias(s, alias) for s in stems):
        return {"corrected_question": q, "corrections": [], "unresolved": []}

    # L3: L2 미해결 토큰만 편집거리 후보 탐색 (그래프=alias 실존 엔티티로 후보 제한, ADR-7).
    corrections: list[dict] = []
    unresolved: list[tuple[str, list[str]]] = []
    out_tokens = list(tokens)

    for i, (tok, stem) in enumerate(zip(tokens, stems)):
        if _in_alias(stem, alias):
            continue
        suffix = tok[len(stem):]  # 제거했던 조사(있다면) 복원용
        # ponytail: O(|alias|) 선형 스캔 — alias가 수천 개면 즉시 끝난다.
        # 수만 개로 커지면 length-bucket 인덱스로 가지치기, 지금은 과설계.
        candidates = sorted(key for key in alias if _edit_distance(stem, key) <= 2)
        close = [k for k in candidates if _edit_distance(stem, k) <= 1]
        if len(close) == 1:
            corrections.append({"from_": stem, "to": close[0], "source": "L3"})
            out_tokens[i] = close[0] + suffix
        else:
            unresolved.append((tok, candidates))

    return {
        "corrected_question": " ".join(out_tokens),
        "corrections": corrections,
        "unresolved": unresolved,
    }


def demo():
    # (a) L1만: NFC 정규화 + 공백 정리, alias 매치 없어도 corrections는 비지만
    #     토큰 자체가 후보 0개(빈 alias)라 unresolved에 그대로 쌓인다 — alias를 준
    #     아래 케이스들과 달리 여기선 빈 alias로 순수 L1 동작만 확인한다.
    r = correct("  베토벤   은   어디서   태어났어 ", {})
    assert r["corrected_question"] == "베토벤 은 어디서 태어났어"
    assert r["corrections"] == []

    alias = {"베토벤": "베토벤", "아인슈타인": "아인슈타인", "함부르크": "함부르크"}

    # (b) L2 조기종료: 정확매치 + 접두매치('아인'은 '아인슈타인'의 prefix) → L3 스킵
    r = correct("베토벤 아인", alias)
    assert r["corrections"] == [] and r["unresolved"] == []
    assert r["corrected_question"] == "베토벤 아인"

    # (c) L3 유일 근접후보(거리<=1) 자동교정: '아인시타인' → '아인슈타인' (거리 1).
    #     '태어났어'는 alias에 없는 일반 어휘라 후보 0개로 unresolved에 남는다 —
    #     의도된 동작(L3는 alias=그래프 엔티티만 대상, 엄한 후보 제한 ADR-7).
    r = correct("아인시타인이 함부르크에서 태어났어", alias)
    assert {"from_": "아인시타인", "to": "아인슈타인", "source": "L3"} in r["corrections"]
    assert "아인슈타인이" in r["corrected_question"] and "함부르크에서" in r["corrected_question"]
    assert not any(tok == "아인시타인이" for tok, _ in r["unresolved"])

    # (d) L3 애매(후보 복수, 거리 2 동률) → 추측 금지, 후보 목록과 함께 unresolved로 위임
    ambiguous_alias = {"본": "본", "빈": "빈", "베토벤": "베토벤"}
    r = correct("전이 어디야", ambiguous_alias)
    assert r["corrections"] == []
    assert any(tok == "전이" for tok, _ in r["unresolved"])
    cands = dict(r["unresolved"])["전이"]
    assert set(cands) >= {"본", "빈"}

    print("correct demo OK")


if __name__ == "__main__":
    demo()
