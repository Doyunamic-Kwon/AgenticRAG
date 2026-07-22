# 04. WBS (14일)

표기: [W#.#] 작업ID / (의존) / → 산출물 / ✓ 완료 기준

## W0. 사전 검증 (Day 0 — 오늘, 다른 모든 것보다 먼저)
- [W0.1] KorQuAD 수율 체크 스크립트 (07_첫번째_태스크.md 상세 스펙) / (없음)
  → bridge_candidates.jsonl, relation_stats.json, corpus_doc_list.txt
  ✓ 브릿지 후보 ≥ 300쌍 (200문항 목표의 1.5배 여유). 미달 시 KorQuAD 2.0 확장 결정
- [W0.2] relation 스키마 10종 확정 (W0.1의 분포 결과로) → configs/relations.yaml 확정
- [W0.3] 인간 확인: API 크레딧 한도, 팀 분담, 발표일 (병행, 블로킹 아님)

## W1. 데이터 기반 (Day 1~2)
- [W1.1] 코퍼스 확보: W0.1의 corpus_doc_list(평가 문서+하이퍼링크 이웃) 위키 덤프에서 추출·청킹 / (W0.1)
  → data/corpus.jsonl ✓ 문단 수 상한 3만 이내, 평가 문항 정답 문서 포함률 100%
- [W1.2] Chroma 인덱싱 + 한국어 검색 품질 스팟체크 / (W1.1)
  ✓ 샘플 20질문 Recall@10 목측 통과. 미달 시 한국어 특화 임베딩(예: KoE5 계열)으로 교체 결정
- [W1.3] 트리플 추출 배치 (gpt-4o-mini, relations.yaml 고정 프롬프트, 캐싱) / (W1.1, W0.2)
  → NetworkX pickle (엣지에 source_paragraph_id) ✓ 샘플 100엣지 수검수 정확도 ≥85%
- [W1.4] alias 사전 (위키 redirect + 문서 제목) / (W1.1) → data/alias.json
- [W1.5] 평가 하네스 골격 + **Baseline 측정** / (W1.2) ✓ ★게이트: Baseline Recall@k 수치 확보
- [W1.6] 멀티홉 평가셋 구축 착수 (브릿지 쌍 → 질문 합성·검수, 05_평가_명세 스키마 준수) / (W0.1)

## W2. 코어 모듈 (Day 3~4)
- [W2.1] 뼈대: configs 로더, models.py(명세 §0), trace / (없음)
- [W2.2] infra: llm.py(재시도·캐싱·비용집계 단일 관문), embeddings, vector_store, graph_store(GraphStore Protocol+NetworkX) / (W1.3)
- [W2.3] 교정 [0] L1~L3 / (W1.4) ✓ 오타 샘플 20개 중 L3까지 회수 ≥70%
- [W2.4] 분해 [1] + 계획 검증 8종 [2] / (W2.1) ✓ 시나리오 A/B/C 질문의 분해 JSON 수검수 통과
- [W2.5] 그라운딩 [3] / (W2.2) ✓ exact/스냅/타이브레이크/포기 4경로 단위테스트
- [W2.6] 실행기 + resolve_hop 골격(검증 없이 검색만) / (W2.4, W2.5) ✓ ★게이트: 2-hop E2E 동작

## W3. 검증·재질의 (Day 5~6) — 프로젝트의 심장
- [W3.1] Quad Verification [5] + 병합 규칙 + shortcut / (W2.6) ✓ 시나리오 A 재현
- [W3.2] 메모이제이션 + verifier 주입 구조 / (W3.1) — 지금 안 넣으면 나중에 못 넣음 (ADR-4)
- [W3.3] 힌트 재질의 [6] + 탈출 경로 / (W3.1)
- [W3.4] final_op [7] + Path-Check [8] + 응답 조립 [9] / (W3.1)
  ✓ ★게이트: 시나리오 B 재현 (빈 backlink 탈락 → 재질의 → 잘츠부르크)

## W4. 대조군·평가 (Day 7~8)
- [W4.1] Agent-basic (llm_judge verifier 주입) + Baseline 모드 스위치 / (W3.2)
- [W4.2] 멀티홉 평가셋 완성(100~200) + 오타 변형 30 / (W1.6) ✓ shortcut 문항 제거 필터 통과
- [W4.3] **1차 3단 비교 + oracle@1/@2 분석** / (W4.1, W4.2)
  ✓ ★게이트: Ours > Agent-basic. 미달 시 Day 9~10 전량 검증 개선 투입
  ✓ oracle 갭으로 조건부 분기(ADR-8) 도입 여부 결정

## W5. 데모·개선 (Day 9~11)
- [W5.1] (W4.3 결정에 따라) 조건부 분기 구현 or 검증 개선
- [W5.2] Streamlit: 계획·hop trace·검증 뱃지·재작성 질의·반복 횟수·근거·출처 (R6 전부) + 비교 토글
  ✓ 외부인 데모 테스트 통과
- [W5.3] 그래프·alias 보정 → 2차 평가 ✓ ★게이트: 수치 동결

## W6. 마감 (Day 12~14)
- [W6.1] (옵션) STAR 확대·배칭·Fast-Path
- [W6.2] 발표 자료: 문제 라이브 → Agent-basic 오답 통과 라이브 → Ours 차단·회복 라이브 → 3단 숫자 → 설계 인과 → 정직한 한계
- [W6.3] README·기술문서·재생성 스크립트 정리, 리허설 2회, 제출

## 크리티컬 패스
W0.1 → W1.1 → W1.2 → W1.5(게이트) → W2.6(게이트) → W3.4(게이트) → W4.3(게이트) → W5.3(동결) → W6.3
지연 시 버리는 순서: W6.1 → W5.2 비교토글 → STAR → (최후) 멀티홉 셋 규모 축소(100 유지)
