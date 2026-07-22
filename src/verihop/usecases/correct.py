"""[0] 교정 L1~L3 (결정적, LLM 금지)
reads:  raw_question, alias·약어 사전
writes: corrected(잠정), corrections, 미해결 토큰+후보
may-import: verihop.models, verihop.ports (EmbedderPort — 자모/편집거리 후보용)
test: tests/usecase/test_correct.py (fakes 주입)
done-check: tools/check_layers.sh 통과
"""
# W2.3. usecase는 함수(클래스 아님). ports만 인자로 받는다.
raise NotImplementedError
