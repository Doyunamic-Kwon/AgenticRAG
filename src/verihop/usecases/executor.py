"""ready-set 루프 + tie 분기 (재귀 함수 금지, ADR-3)
reads:  hops[], execution_order
writes: hop_results, 체인 분기
may-import: verihop.models, verihop.usecases.resolve_hop
test: tests/usecase/test_executor.py (fakes 주입)
done-check: tools/check_layers.sh 통과
"""
# W2.6. usecase는 함수(클래스 아님). ports만 인자로 받는다.
raise NotImplementedError
