"""[7]final_op + [8]Path-Check·체인중재 + [9]응답조립
reads:  hop_results, final_op
writes: Answer, path_check
may-import: verihop.models, verihop.ports (LLMPort, GraphPort)
test: tests/usecase/test_finalize.py (fakes 주입)
done-check: tools/check_layers.sh 통과
"""
# W3.4. usecase는 함수(클래스 아님). ports만 인자로 받는다.
raise NotImplementedError
