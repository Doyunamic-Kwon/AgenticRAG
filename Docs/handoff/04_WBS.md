# 04. WBS (14일)

각 작업은 작업ID [W#.#], 의존 작업, 산출물, 완료 기준을 함께 적는다.

**진행상황 표기 규칙**: 각 항목 아래 `▸ 상태` 줄에 완료/진행중/미착수와 핵심 수치·특이사항을 적는다.
상세 경위(무엇을 시도했고 무엇이 왜 실패했는지)는 항목마다 반복하지 않고 `Docs/진행기록.md`를
가리킨다 — WBS는 결론만, 진행기록은 과정 전체를 담는 분리 원칙(본 프로젝트 문서 구조 원칙과 동일).
이 표는 2026-07-24 기준 스냅샷이다.

## W0. 사전 검증 (Day 0, 오늘)

다른 모든 작업보다 먼저 끝낸다.

- [W0.1] KorQuAD 수율 체크 스크립트. 상세 스펙은 07_첫번째_태스크.md에 있고, 의존 작업은 없다. 산출물은 bridge_candidates.jsonl, relation_stats.json, corpus_doc_list.txt다. 브릿지 후보가 300쌍 이상이면 통과하며, 이는 200문항 목표의 1.5배 여유다. 미달하면 KorQuAD 2.0 확장을 결정한다.
  ▸ **완료 — 미달로 경로 전환.** KorQuAD 브릿지 수율이 목표에 못 미쳐 멀티홉 평가셋을 2WikiMultihopQA 적응 경로(W1.6)로 전환. 원인·경위는 진행기록.md W1.6절 참조.
- [W0.2] relation 스키마 10종을 확정한다(W0.1의 분포 결과를 근거로). 산출물은 configs/relations.yaml.
  ▸ **완료 + 확장.** 초기 10종에 이후 hostsEvent(시나리오 B 대응)·EVENT 타입 추가. 현재 configs/relations.yaml 기준.
- [W0.3] 사람이 확인할 것: API 크레딧 한도, 팀 분담, 발표일. 다른 작업과 병행하며 블로킹은 아니다.
  ▸ 사용자 확인 사항 — 별도 추적 안 함.

## W1. 데이터 기반 (Day 1~2)

- [W1.1] 코퍼스 확보(의존: W0.1). W0.1의 corpus_doc_list, 즉 평가 문서와 하이퍼링크 이웃을 위키 덤프에서 추출하고 청킹한다. 산출물은 data/corpus.jsonl. 문단 수는 3만 이내로 유지하고 평가 문항 정답 문서 포함률은 100%여야 통과한다.
  ▸ **완료 + 증분 확장.** data/corpus.jsonl(KorQuAD 기반) + data/corpus_2wiki.jsonl(2Wiki 적응, 현재 251문단) + data/corpus_demo.jsonl(시연용 5개 문서). 총 임베딩 문단 10,871개(2026-07-24 기준).
- [W1.2] Chroma 인덱싱과 한국어 검색 품질 스팟체크(의존: W1.1). 샘플 20질문의 Recall@10을 눈으로 확인해 통과 여부를 본다. 미달하면 한국어 특화 임베딩(예: KoE5 계열)으로 교체를 결정한다.
  ▸ **완료 — 벡터 백엔드는 계획 변경(ADR-11).** Chroma 대신 FAISS flat + Upstage solar-embedding-1-large(dim 4096) 채택(근거: ADR-11, 코퍼스 규모상 ANN 불필요). 127문항 확대셋 기준 baseline Recall@5=40.2%, @10=48.8%, MRR=0.289, nDCG@10=0.337(2026-07-24 재측정, 이전 46문항 셋 수치와 직접 비교 불가 — 표본 분포가 다름).
- [W1.3] 트리플 추출 배치(의존: W1.1, W0.2). gpt-4o-mini로 relations.yaml 고정 프롬프트와 캐싱을 써서 돌린다. 산출물은 NetworkX pickle이고, 엣지에 source_paragraph_id를 담는다. 샘플 100엣지를 손으로 검수해 정확도가 85% 이상이면 통과한다.
  ▸ **완료, 반복 보정 포함.** LLM은 Upstage solar-pro(계획 대비 provider 변경). 코퍼스 전체로 22.8배 확대(339→7,716→정리 후 노드), 확대 직후 정확도가 87%→70%로 하락해 재판정 기반 프루닝(Track 1, ADR 별도 문서화)으로 98%까지 회복. 2026-07-24 재추출로 현재 노드 6,307 · 엣지 5,428. 대명사·일반 범주어(그룹/사람/책/소설/영화/밴드) 오염 발견·차단(ADR-12 연계).
- [W1.4] alias 사전 구축(의존: W1.1). 위키 redirect와 문서 제목을 모은다. 산출물은 data/alias.json.
  ▸ **완료.** 현재 8,123개(범주어 6개 제외 후). MANUAL_ALIASES로 표기 변이 보강(비엔나→빈 등).
- [W1.5] 평가 하네스 골격과 Baseline 측정(의존: W1.2). **게이트**: Baseline Recall@k 수치를 확보한다.
  ▸ **게이트 통과(Day 2).** eval/run_eval.py --mode baseline. 최신 수치는 W1.2 항목 참조.
- [W1.6] 멀티홉 평가셋 구축에 착수한다(의존: W0.1). 브릿지 쌍에서 질문을 합성하고 검수하며, 05_평가_명세의 스키마를 따른다.
  ▸ **완료 + 계획 대비 확대.** KorQuAD 브릿지 대신 2WikiMultihopQA(scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35) validation split(12,576행) 전량 스캔해 자체 스키마로 적응. 46개(1차) → 127개(2026-07-24, W4.2 목표 100~200 충족)로 확대. 과정에서 fetch_rows 조기중단·KeyError·gold answer_ko 오염(LLM이 정답 미제공 상태로 추측해 다리 엔티티 이름을 답으로 오기입, 4건 확인) 등 버그 다수 발견·수정 — 상세는 진행기록.md 2026-07-24절.

## W2. 코어 모듈 (Day 3~4)

- [W2.1] 뼈대 잡기(의존 없음). settings.py는 configs 로더, models.py는 §0, ports.py는 Protocol 5종, domain/은 plan_rules·verify_rules 순수 함수를 담는다.
  ▸ **완료.** ADR-10 4계층 구조(domain←usecases←adapters←bootstrap/apps/scripts)로 구현, tools/check_layers.sh로 경계 자동 검사.
- [W2.2] adapters 구현(의존: W1.3). openai_llm.py는 재시도·캐싱·비용 집계를 한곳에서 처리하는 관문이고, 그 밖에 openai_embedder, chroma_index, networkx_graph(GraphPort와 NetworkX), json_tracer, memo_cache를 만든다.
  ▸ **완료(벡터 어댑터는 faiss_index로 변경, ADR-11).** openai_llm/openai_embedder/faiss_index/networkx_graph/json_tracer 구현. LLM 재시도 예산은 2026-07-24 429 다발 실측 후 2→6으로 상향(configs/settings.yaml).
- [W2.3] 교정 [0]의 L1~L3(의존: W1.4). 오타 샘플 20개 중 L3까지 회수율이 70% 이상이면 통과한다.
  ▸ **완료.** usecases/correct.py, 결정적 구현(LLM 미사용). EM 채점용 조사 정규화(_strip_particle)도 여기서 재사용.
- [W2.4] 분해 [1]과 계획 검증 8종 [2](의존: W2.1). 시나리오 A/B/C 질문의 분해 JSON을 손으로 검수해 통과하면 된다.
  ▸ **완료, 다수 보정.** usecases/decompose.py + domain/plan_rules.py. 목표타입 추론("A의 B" 마지막 '의' 뒤만 보도록 수정), 규칙 9(이미 이름으로 등장한 엔티티는 선행 hop 지어내지 않기) 추가. 2026-07-24: 계획검증 8번(정답 누출)이 alias 전체를 오탐해 SIMPLE 폴백률 92.9%까지 치솟은 것을 발견·수정(ADR-12) — 8.7%로 개선.
- [W2.5] 그라운딩 [3](의존: W2.2). exact, 스냅, 타이브레이크, 포기 4경로를 단위테스트로 검증한다.
  ▸ **완료.** usecases/ground.py. 그라운딩을 hop 실행 시점으로 옮겨 참조 hop의 start_node가 영구 None이 되던 버그 수정(bootstrap._rh).
- [W2.6] 실행기와 resolve_hop 골격(의존: W2.4, W2.5). 이 단계에서는 검증 없이 검색만 한다. **게이트**: 2-hop E2E가 동작한다.
  ▸ **게이트 통과.** usecases/executor.py(ready-set 루프) + usecases/resolve_hop.py.

## W3. 검증·재질의 (Day 5~6)

이 프로젝트에서 가장 중요한 구간이다.

- [W3.1] Quad Verification [5], 병합 규칙, shortcut(의존: W2.6). 시나리오 A를 재현하면 통과한다.
  ▸ **완료.** domain/verify_rules.py. backlink 3-값(pass/absent/contradict) 하드 비토 — contradict면 pass_ratio 무관하게 탈락(ADR-6: 원래 의도한 hop 관계로 검사, 그라운딩된 관계 아님).
- [W3.2] 메모이제이션과 verifier 주입 구조(의존: W3.1). 지금 넣지 않으면 나중에 넣기 어렵다(ADR-4).
  ▸ **완료.** [4]검색 결과에만 메모이제이션 범위 한정(ADR-4/C6) — 검증기 선택·hop_id·재시도·FAILED 상태는 캐시 안 함(4-arm 비교 오염 방지).
- [W3.3] 힌트 재질의 [6]와 탈출 경로(의존: W3.1).
  ▸ **완료.** resolve_hop.py REQUERY_PROMPT. 2026-07-24: 엔티티 추출 프롬프트에 조사 제거 지시 추가(alias 매칭 실패 방지).
- [W3.4] final_op [7], Path-Check [8], 응답 조립 [9](의존: W3.1). **게이트**: 시나리오 B를 재현한다. 빈 backlink로 탈락한 뒤 재질의를 거쳐 잘츠부르크에 도달하는 흐름이다.
  ▸ **게이트 통과.** usecases/finalize.py(결정적 템플릿 조립, LLM 미사용). 시나리오 B 재현 중 원인 3건 발견·수정(목표타입 오추론, 대명사 "그"의 alias 오염, 동시추출 비결정성으로 인한 엔티티 타입 충돌) — 경위는 진행기록.md 참조.

## W4. 대조군·평가 (Day 7~8)

- [W4.1] Agent-basic(llm_judge verifier 주입)과 Baseline 모드 스위치(의존: W3.2).
  ▸ **완료.** bootstrap.build_pipeline(mode)의 mode 분기(baseline/agent_basic/ours_g/ours). Ours-G(분해+그라운딩은 Ours와 동일하되 검증만 llm_judge)를 추가해 "분해 기여 vs 검증 기여"를 분리(계획 대비 확장, C4).
- [W4.2] 멀티홉 평가셋 완성(100~200)과 오타 변형 30(의존: W1.6). shortcut 문항 제거 필터를 통과해야 한다.
  ▸ **부분 완료.** 멀티홉 127문항 확보(100~200 범위 충족, W1.6 참조) — 다만 2Wiki 검증 split 전량(12,576행)을 스캔한 상한이라 150 목표는 못 채움(현재 필터 기준 자연 한계). shortcut 필터는 2Wiki 셋에 **아직 미적용**(baseline 수치가 필터 전). 오타 변형 30문항은 **미착수**.
- [W4.3] 1차 3단 비교와 oracle@1/@2 분석(의존: W4.1, W4.2). **게이트**: Ours가 Agent-basic보다 낫다. 미달하면 Day 9~10을 전량 검증 개선에 투입한다. oracle 갭을 보고 조건부 분기(ADR-8) 도입 여부를 결정한다.
  ▸ **진행중(게이트 미확정) — 2026-07-24 대규모 재작업 중.** 46문항 셋에서는 EM 기준 Ours≈Agent-basic(동률)이었으나 "고신뢰오답"(05 §4 KPI, ANSWERED인데 오답)은 Ours가 더 적어 검증 효과가 존재함을 시사. 127문항으로 확대 후 재실행하니 decompose SIMPLE 폴백 92.9%로 3개 arm 모두 EM 3~4%까지 붕괴 — 원인 수정(ADR-12) 후 재실행이 429 레이트리밋과 겹쳐 현재 재시도로 수렴 중. oracle 분석은 미착수. 최종 결론은 진행기록.md에 확정 시 기록.

## W5. 데모·개선 (Day 9~11)

- [W5.1] W4.3 결정에 따라 조건부 분기를 구현하거나 검증을 개선한다.
  ▸ **미착수.** W4.3 게이트 확정 후 판단.
- [W5.2] Streamlit UI. 계획, hop trace, 검증 뱃지, 재작성 질의, 반복 횟수, 근거, 출처를 모두 보여준다(R6 전부). 비교 토글도 넣는다. 외부인 데모 테스트를 통과하면 된다.
  ▸ **완료.** apps/streamlit_app.py — 시나리오 A/B/C 프리셋, hop별 검증뱃지, plain_rag(일반 RAG) 비교 패널. streamlit.testing.v1.AppTest로 시나리오 B 무예외 검증(실제 브라우저 수동 확인은 미실시).
- [W5.3] 그래프와 alias를 보정한 뒤 2차 평가를 돌린다. **게이트**: 수치를 동결한다.
  ▸ **진행중(동결 전).** 그래프·alias 보정은 여러 라운드 진행(범주어 정리, occupation 오탐 수정 등). 2차 평가(127문항 4-arm)는 W4.3과 동시 진행 중 — 완료 후 수치 동결 예정.

## W6. 마감 (Day 12~14)

- [W6.1] (옵션) STAR 확대, 배칭, Fast-Path.
  ▸ 미착수.
- [W6.2] 발표 자료. 문제를 라이브로 보여주고, Agent-basic이 오답을 통과시키는 장면을 라이브로 보여준 다음, Ours가 이를 차단하고 회복하는 장면을 라이브로 보여준다. 이어서 3단 숫자, 설계의 인과, 정직한 한계를 차례로 다룬다.
  ▸ 미착수. backlink 비토 메커니즘은 실제 그래프 데이터로 직접 검증 완료(_backlink_state 호출 시연 가능 — 모차르트→빈=contradict, →잘츠부르크=pass).
- [W6.3] README와 기술문서, 재생성 스크립트를 정리하고, 리허설을 2회 한 뒤 제출한다.
  ▸ 미착수.

## 크리티컬 패스

W0.1 → W1.1 → W1.2 → W1.5(게이트 ✅) → W2.6(게이트 ✅) → W3.4(게이트 ✅) → W4.3(게이트 진행중, 미확정) → W5.3(동결 대기) → W6.3

지연되면 버리는 순서는 W6.1, W5.2의 비교 토글, STAR, 마지막으로 멀티홉 셋 규모 축소(100 유지) 순이다.
