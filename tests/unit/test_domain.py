"""domain 순수 규칙 단위테스트 (mock 없음). 각 모듈의 demo() = 계약 자체검증."""
from verihop.domain import plan_rules, verify_rules


def test_verify_rules():
    verify_rules.demo()


def test_plan_rules():
    plan_rules.demo()
