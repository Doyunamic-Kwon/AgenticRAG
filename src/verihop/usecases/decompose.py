"""[1] 분해 통합 콜 (LLM 1콜, structured output)
reads:  corrected(잠정), 미해결 토큰+후보, relation 스키마
writes: hops[], final_op, goal_type_surface, corrected(확정), corrections(L4)
may-import: verihop.models, verihop.ports (LLMPort)
test: tests/usecase/test_decompose.py (fakes 주입)
done-check: tools/check_layers.sh 통과
"""
# W2.4. usecase는 함수(클래스 아님). ports만 인자로 받는다.
raise NotImplementedError
