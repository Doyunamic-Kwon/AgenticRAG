"""조립 루트. build_pipeline(mode)가 유일한 조립 지점 (ADR-10).
Settings 로드 → adapters 생성 → usecases에 주입. DI 프레임워크 없음(평범한 함수).
mode: FULL(verifier=quad) | SIMPLE | agent_basic(분해 없음, llm_judge) | ours_g(FULL + llm_judge) | baseline
test: apps/cli.py 로 스모크
"""
# W2.1+
raise NotImplementedError
