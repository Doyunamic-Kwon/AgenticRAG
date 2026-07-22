"""[4] 경로A 검색결과 캐시 데코레이터 (C6).
캐시 대상은 검색결과(Candidate 목록)만 — 검증·재질의는 매번 재실행.
키: (start_resolved, relation_grounded, direction, sub_query, definition).
verifier·hop_id·retries·FAILED를 키/값에 넣지 않는다(3단 비교·재평가 오염 방지).
스코프: 프로세스 내 메모리. 평가 run 간 비공유.
test: tests/usecase/test_memo_cache.py
"""
# W3.2
raise NotImplementedError
