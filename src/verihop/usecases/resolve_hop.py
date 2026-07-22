"""[4]검색+[5]검증확정+[6]재질의 캡슐화 = 원자 연산 (ADR-3/4)
reads:  Hop, state, verifier
writes: HopResult (candidates 전량 보존)
may-import: verihop.models, verihop.ports (LLM/Vector/Graph), verihop.domain.verify_rules
캐시: [4] 검색결과만 (memo_cache), 검증·재질의는 매번 재실행 (C6)\nverifier 주입: quad(domain.verify_rules) | llm_judge(Ours−G arm, C4)\ntest: tests/usecase/test_resolve_hop.py (fakes 주입)
done-check: tools/check_layers.sh 통과
"""
# W2.6/W3.1. usecase는 함수(클래스 아님). ports만 인자로 받는다.
raise NotImplementedError
