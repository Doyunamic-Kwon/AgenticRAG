# data/ 레이아웃

이 디렉토리 내용은 git에서 제외한다(용량·라이선스). 구조는 이 README로만 기록한다.
경로는 configs/settings.yaml의 paths와 05_평가_명세를 따른다.

## 입력 (사람이 배치)
- `raw/` — KorQuAD 1.0 train+dev JSON (korquad.github.io). W0.1 입력.

## W0.1 출력 (scripts/w0_yield_check.py)
- `w0/bridge_candidates.jsonl`
- `w0/relation_stats.json`
- `w0/corpus_doc_list.txt`
- `w0/report.md`

## W1 산출
- `corpus.jsonl` — 청킹된 코퍼스 (W1.1)
- `chroma/` — Chroma 인덱스 (W1.2)
- `graph.pkl` — NetworkX pickle, 엣지에 source_paragraph_id (W1.3)
- `alias.json` — alias 사전 (W1.4)
- `eval/multihop.jsonl` — 멀티홉 gold 평가셋 (W1.6/W4.2, 05 gold 스키마)

평가 런 출력은 repo 루트의 `results/{run_id}/`에 쌓인다 (05 §5, git 제외).
