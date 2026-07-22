"""Quad 검증 판정 + passed 산식 — 순수 함수, I/O 없음 (03 [5], C1).
검사 4종(type/name/desc/backlink)의 결과를 받아 Verification을 만든다. 임베딩·그래프 조회는
호출측(usecases/resolve_hop)이 수행하고 결과(bool / cos값 / BacklinkState)만 넘긴다.
passed 산식 (C1):
  1. backlink == "contradict" → passed=False (hard-veto, pass_ratio 무관)
  2. 그 외: "absent"·None(약화모드)은 시행수 제외(중립), 나머지 시행 검사 통과율 ≥ pass_ratio
may-import: verihop.models
test: tests/unit/test_verify_rules.py
"""
from verihop.models import BacklinkState, Verification


def decide(type_ok: bool | None, name_ok: bool | None, desc_ok: bool,
           backlink: BacklinkState | None, pass_ratio: float, weak_mode: bool) -> Verification:
    v: Verification = {
        "type_check": type_ok, "name_check": name_ok, "desc_check": desc_ok,
        "backlink_check": backlink, "passed": False, "weak_mode": weak_mode,
    }
    if backlink == "contradict":                 # C1: 하드 veto (pass_ratio 무관)
        return v
    shipped: list[bool] = []
    if type_ok is not None:
        shipped.append(bool(type_ok))
    if name_ok is not None:
        shipped.append(bool(name_ok))
    shipped.append(bool(desc_ok))                # desc는 항상 시행
    if backlink == "pass":
        shipped.append(True)
    # backlink "absent"·None → 시행수에서 제외 (중립)
    v["passed"] = bool(shipped) and sum(shipped) / len(shipped) >= pass_ratio
    return v


def demo():
    R = 0.75
    # 정상 4/4 통과
    assert decide(True, True, True, "pass", R, False)["passed"]
    # 시나리오 B (빈): type✓ name✓ desc✓ backlink=contradict → veto (핵심 C1)
    v = decide(True, True, True, "contradict", R, False)
    assert v["passed"] is False and v["backlink_check"] == "contradict"
    # backlink absent(중립): type·name·desc 3/3 → 통과
    assert decide(True, True, True, "absent", R, False)["passed"]
    # 3/4 정확히 0.75 (type✓ name✗ desc✓ backlink=pass) → 통과
    assert decide(True, False, True, "pass", R, False)["passed"]
    # 2/4=0.5 < 0.75 → 실패
    assert not decide(False, False, True, "pass", R, False)["passed"]
    # 약화모드(graph_miss): desc만 시행 → 1/1=100% 통과 (finding #9: weak desc-only, 명세대로)
    assert decide(None, None, True, None, R, True)["passed"]
    print("verify_rules demo OK")


if __name__ == "__main__":
    demo()
