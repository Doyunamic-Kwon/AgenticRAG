"""seam Protocol 5개 — usecase 계층이 보는 유일한 외부 계약 (ADR-10, C4).
usecase는 이 Protocol만 안다. 구체 구현(adapters/*)의 존재를 모른다.
상한 5개: 6번째 포트를 추가하려면 02에 ADR을 먼저 쓴다."""
from __future__ import annotations
from typing import Optional, Protocol
from verihop.models import Direction, TraceEvent


class LLMPort(Protocol):
    def complete(self, prompt: str, *, schema: Optional[dict] = None) -> object:
        """structured output(schema 지정 시 dict). 재시도·캐싱·비용집계는 구현 내부.
        전 LLM 콜이 이 관문을 지난다 → 콜수·비용 지표 자동 산출."""
        ...


class EmbedderPort(Protocol):
    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """임베딩 + 캐시. 코사인 계산은 호출측(usecase)에서.
        is_query: solar 등 문서/질의 임베딩 모델이 다른 경우 분기."""
        ...


class VectorIndexPort(Protocol):
    def search(self, query: str, top_k: int) -> list[tuple[str, str, float]]:
        """경로A. [(paragraph_id, text, score)] 상위 top_k (≥10, C3)."""
        ...


class GraphPort(Protocol):
    def neighbors(self, node: str, relation: str, direction: Direction) -> list[tuple[str, list[str]]]:
        """경로B. [(neighbor_node, evidence_paragraph_ids)]."""
        ...

    def has_edge(self, src: str, dst: str, relation: str) -> bool:
        """backlink 검사용 (C1)."""
        ...

    def node_type(self, node: str) -> Optional[str]:
        """type_check용. 미존재/미태깅 → None."""
        ...

    def relations_of(self, node: str) -> list[str]:
        """그라운딩 포기 시 hint_relations / 재질의 힌트."""
        ...

    def exists(self, node: str) -> bool: ...


class TracerPort(Protocol):
    def emit(self, event: TraceEvent) -> None:
        """구조화 로그. request_id 관통 (03 [로깅])."""
        ...
