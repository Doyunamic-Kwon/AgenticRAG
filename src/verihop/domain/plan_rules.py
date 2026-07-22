"""계획 검증 8종 — 순수 함수, I/O 없음 (03 [2]). mock 없이 단위테스트.
reads:  Hop[], FinalOp, goal_type_surface
writes: 검사별 pass/fail + 실패 처리 지시(재생성/SIMPLE/절단/final_op 제거)
may-import: verihop.models (그 외 verihop·서드파티 import 금지)
검사 8: definition 내 '그 hop 답(tail)·후속 참조 엔티티'명 검출. 앵커·조건 서술의 기존 엔티티는 허용 (C5)
test: tests/unit/test_plan_rules.py
"""
from verihop.models import Hop, FinalOp


def validate_plan(hops: list[Hop], final_op: FinalOp | None, goal_type_surface: str | None):
    raise NotImplementedError  # W2.4
