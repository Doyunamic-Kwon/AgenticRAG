"""일반 RAG 대조군 — 임베딩 유사도 검색 → LLM이 바로 답변. 분해·그래프·검증·재질의 전혀 없음.

시연 대시보드에서 VeriHop(Ours)과 나란히 두고 차이를 보여주기 위한 최소 구현. bootstrap의 어댑터를
재사용해 같은 코퍼스·같은 임베딩/LLM 모델로 검색하므로, 눈에 보이는 차이는 순수하게 에이전틱 레이어
(분해·그래프 검증·재질의) 유무뿐이다 — 05 평가 명세의 공정성 원칙과 같은 논리.

usecases/가 아니라 최상위 모듈로 둔 이유: 이 파일은 의도적으로 VeriHop 파이프라인 계약(ports/도메인
규칙)을 우회한다 — 비교 대조군이지 시스템의 일부가 아니다. bootstrap.py처럼 adapters를 직접 쓴다.
"""
from __future__ import annotations
from verihop.bootstrap import build_adapters, _load_settings

ANSWER_PROMPT = """아래 문단들만 근거로 질문에 답하라. 문단에 근거가 없으면 "모르겠다"고 답하라.
과정·근거 설명 없이 답만 간결하게 한국어로 말하라.

문단:
{paragraphs}

질문: {question}"""


def answer(question: str, *, adapters=None, settings=None, top_k: int = 10) -> dict:
    """returns {text, evidence_paragraph_ids}. 검색 1회 + LLM 답변 1콜, 그 이상 없음."""
    settings = settings or _load_settings()
    adapters = adapters or build_adapters(settings)
    hits = adapters["vector"].search(question, top_k)          # [(pid, text, score)]
    paras_block = "\n\n".join(f"[{pid}] {text[:800]}" for pid, text, _ in hits)
    text = adapters["llm"].complete(ANSWER_PROMPT.format(paragraphs=paras_block, question=question))
    return {"text": (text or "").strip(), "evidence_paragraph_ids": [pid for pid, _, _ in hits[:3]]}
