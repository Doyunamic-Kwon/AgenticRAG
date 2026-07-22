"""§0 공용 데이터 객체 — 파이프라인 전 모듈의 유일한 데이터 계약 (03_파이프라인_명세 §0).
domain 계층: 표준 라이브러리만 import. 모든 에이전트가 이 타입만 주고받는다."""
from __future__ import annotations
from typing import Literal, Optional, TypedDict

Type = Literal["PERSON", "LOCATION", "ORG", "DATE", "NUMBER", "WORK", "TERM"]
Direction = Literal["fwd", "inv"]
BacklinkState = Literal["pass", "absent", "contradict"]  # C1: 3값


class Correction(TypedDict):
    from_: str          # 'from'은 예약어 → from_
    to: str
    source: Literal["L1", "L2", "L3", "L4"]


class Expected(TypedDict):
    type: Type
    definition: str     # 조건의 재서술. 정답 고유명사 금지 (계획검증 8, C5)


class AnchorMeta(TypedDict):
    name: str
    disambiguator: Optional[str]


class Hop(TypedDict):
    id: int
    start: str                        # 리터럴 or "{n}" 참조
    start_node: Optional[str]         # 그라운딩 후 노드 ID
    relation: str                     # 스키마 10종 + "relatedTo"
    relation_grounded: Optional[str]
    direction: Direction
    expected: Expected
    anchor_meta: AnchorMeta
    sub_query: str
    hint_relations: Optional[list[str]]


class FinalOp(TypedDict):
    operator: Literal["EARLIER", "LATER", "SAME", "GREATER", "SMALLER"]
    in_: list[int]                    # 'in'은 예약어 → in_


class Verification(TypedDict):
    type_check: Optional[bool]        # None = 약화모드 미시행
    name_check: Optional[bool]
    desc_check: bool
    backlink_check: Optional[BacklinkState]   # C1
    passed: bool                      # contradict → False veto, 그 외 시행 통과율 ≥ pass_ratio
    weak_mode: bool


class Candidate(TypedDict):
    entity: str
    origin: Literal["vector", "graph", "both"]
    evidence_paragraph_ids: list[str]
    verification: Optional[Verification]
    score: float


class HopResult(TypedDict):
    hop_id: int
    answer: str
    candidates: list[Candidate]       # 전량 보존 (ADR-8)
    tie: bool
    branched: bool
    retries: int
    status: Literal["OK", "WEAK", "FAILED"]
    queries_used: list[str]


class Evidence(TypedDict):
    paragraph_id: str
    hop_id: int


class Answer(TypedDict):
    text: str
    status: Literal["ANSWERED", "PARTIAL", "UNVERIFIABLE"]
    confidence: float
    path_check: Optional[bool]
    evidence: list[Evidence]
    verified_until: Optional[int]


# {event, request_id, stage, payload} — json_tracer가 소비. 느슨하게 dict.
TraceEvent = dict


class PipelineState(TypedDict):
    request_id: str
    raw_question: str                 # 불변
    corrected_question: str
    corrections: list[Correction]
    goal_type_surface: Optional[str]  # 어미 판독 타입 (검산 전용)
    hops: list[Hop]
    final_op: Optional[FinalOp]
    hop_results: dict[int, HopResult]
    mode: Literal["FULL", "SIMPLE"]   # 파이프라인 실행 모드. Baseline/Agent-basic/Ours−G는 eval --mode 플래그
    answer: Optional[Answer]
    trace: list[TraceEvent]
