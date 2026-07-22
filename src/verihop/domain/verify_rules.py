"""Quad 검증 판정 + passed 산식 — 순수 함수, I/O 없음 (03 [5], C1).
검사 4종(type/name/desc/backlink)의 결과를 받아 Verification을 만든다. 임베딩·그래프 조회는
호출측(usecases/resolve_hop)이 수행하고 결과(bool / cos값 / BacklinkState)만 넘긴다.
passed 산식 (C1):
  1. backlink == "contradict" → passed=False (hard-veto, pass_ratio 무관)
  2. 그 외: "absent"는 시행수 제외(중립), 나머지 시행 검사 통과율 ≥ pass_ratio
may-import: verihop.models
test: tests/unit/test_verify_rules.py
"""
from verihop.models import BacklinkState, Verification


def decide(type_ok: bool | None, name_ok: bool | None, desc_ok: bool,
           backlink: BacklinkState | None, pass_ratio: float, weak_mode: bool) -> Verification:
    raise NotImplementedError  # W3.1
