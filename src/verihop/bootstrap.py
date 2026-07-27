"""조립 루트. build_pipeline(mode)가 유일한 조립 지점 (ADR-10).
Settings 로드 → adapters 생성 → usecases에 주입. DI 프레임워크 없음(평범한 함수).
mode: FULL(=ours, verifier=quad) | SIMPLE | agent_basic(분해 없음, llm_judge) | ours_g(FULL+llm_judge) | baseline
test: apps/cli.py 로 스모크
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from verihop.usecases.correct import correct
from verihop.usecases.decompose import decompose
from verihop.usecases.ground import ground
from verihop.usecases.executor import execute
from verihop.usecases.resolve_hop import resolve_hop
from verihop.usecases.finalize import finalize
from verihop.domain import verify_rules

ROOT = Path(__file__).resolve().parents[2]
KEY_ENV = {"upstage": "UPSTAGE_API_KEY", "openai": "OPENAI_API_KEY", "naver": "NAVER_CLOVA_API_KEY"}


def _load_settings():
    load_dotenv(ROOT / ".env")
    return yaml.safe_load(open(ROOT / "configs/settings.yaml", encoding="utf-8"))


def _load_relations():
    r = yaml.safe_load(open(ROOT / "configs/relations.yaml", encoding="utf-8"))
    # decompose.py가 기대하는 모양: {name: {"head","tail","ko":[...]}} (프롬프트·ko키워드 프리필터용)
    schema = {x["name"]: {"head": x["head"], "tail": x["tail"], "ko": x.get("ko", [])}
              for x in r["relations"]}
    desc = {x["name"]: f"{x['name']}: {' '.join(x.get('ko', []))}" for x in r["relations"]}
    return schema, desc, r["types"]


def build_adapters(settings, llm_provider=None, tracer=None):
    """provider별 어댑터 생성. llm_provider 오버라이드는 Ours-G/Agent-basic 실험용이 아니라
    제공자 전환용(quad/llm_judge 전환은 verifier 함수로, 09.md ADR-4).
    tracer: 기본은 JsonTracer(메모리 누적)지만, 데모 UI의 실시간 4패널 시각화처럼 이벤트를
    스레드 밖(폴링 루프)으로 즉시 흘려보내야 하면 QueueTracer 등 다른 TracerPort 구현을 주입."""
    from verihop.adapters.openai_llm import OpenAILLM
    from verihop.adapters.openai_embedder import OpenAIEmbedder
    from verihop.adapters.faiss_index import FaissIndex
    from verihop.adapters.networkx_graph import NetworkxGraph
    from verihop.adapters.json_tracer import JsonTracer

    ep = settings["providers"]["embedding"]
    ec = settings["embedding"][ep]
    embedder = OpenAIEmbedder(os.environ[KEY_ENV[ep]], ec["base_url"], ec["model"], ec["dim"],
                              batch=settings["embedding"].get("batch", 32),
                              max_chars=settings["embedding"].get("max_chars", 5000))

    lp = llm_provider or settings["providers"]["llm"]
    lc = settings["llm"][lp]
    llm = OpenAILLM(os.environ[KEY_ENV[lp]], lc["base_url"], lc["model"],
                    settings["llm"].get("temperature", 0.0), settings["llm"].get("max_retries", 2))

    vdir = ROOT / settings["vector"]["dir"]
    vector = FaissIndex(vdir, embedder, [ROOT / "data/corpus.jsonl", ROOT / "data/corpus_2wiki.jsonl",
                                         ROOT / "data/corpus_demo.jsonl"])
    graph = NetworkxGraph(str(ROOT / "data/graph.pkl"))
    tracer = tracer if tracer is not None else JsonTracer()
    alias = json.load(open(ROOT / "data/alias.json", encoding="utf-8"))
    return {"llm": llm, "embedder": embedder, "vector": vector, "graph": graph,
            "tracer": tracer, "alias": alias}


def _leak_check_alias_names(adapters):
    """decompose의 8번 검사(정답 누출 방지)에 넘길 alias 집합. 문제: occupation 관계의 tail 값
    (예: "감독"·"가수"·"작곡가"·"배우")은 여러 인물이 공유하는 정당한 속성값인데, alias.json엔
    다른 고유명사와 똑같이 들어있어 8번 검사가 "그 감독이 태어난 곳" 같은 정상적인 definition조차
    '누출'로 오판 — 127문항 중 92.9%가 SIMPLE로 강등되는 원인이었다. occupation의 tail 값만 alias
    집합에서 제외(그래프·grounding용 alias 자체는 그대로 둠 — 이 함수는 leak-check 전용 뷰)."""
    if "_leak_alias_names" not in adapters:
        graph = adapters["graph"]
        occ_values = {n for n in graph.all_nodes()
                     if "occupation" in graph.relations_of(n) and not graph.neighbors(n, "occupation", "fwd")}
        adapters["_leak_alias_names"] = set(adapters["alias"]) - occ_values
    return adapters["_leak_alias_names"]


def llm_judge_verifier(type_ok, name_ok, desc_ok, backlink, pass_ratio, weak_mode):
    """Ours-G/Agent-basic arm용 확률적 검증기 — 그래프 신호 무시, desc(LLM 임베딩 유사도)만으로 판정
    (ADR-4 verifier 주입). 실제 LLM judge 콜은 비용 상 desc_check 재사용으로 근사(MVP 스코프)."""
    return {"type_check": None, "name_check": None, "desc_check": desc_ok,
            "backlink_check": None, "passed": desc_ok, "weak_mode": True}


async def run_pipeline(raw_question, adapters, settings, schema, relation_desc, types,
                       *, mode="FULL", verifier=None):
    """질문 1개를 전 단계로 실행 (03 §1 전체). mode=FULL(Ours) 기준.
    mode=SIMPLE: [0]→경로A 1회→[9]. mode=agent_basic: 분해 없이 원질의로 단일 hop resolve.
    verifier: None이면 quad(domain.verify_rules.decide). Ours-G/Agent-basic은 llm_judge_verifier 전달."""
    verifier = verifier or verify_rules.decide
    alias = adapters["alias"]
    tracer = adapters["tracer"]

    corr = correct(raw_question, alias)

    if mode == "agent_basic":
        # 분해 없음: 원질의를 단일 pseudo-hop으로 resolve (05 §1, C4)
        pseudo_hop = {"id": 1, "start": corr["corrected_question"], "start_node": None,
                      "relation": "relatedTo", "relation_grounded": None, "direction": "fwd",
                      "expected": {"type": "", "definition": corr["corrected_question"]},
                      "anchor_meta": {"name": None, "disambiguator": None},
                      "sub_query": corr["corrected_question"], "hint_relations": None}
        tracer.emit({"type": "plan_ready", "hops": [pseudo_hop], "mode": "agent_basic"})
        state = _state(adapters, settings, relation_desc)
        r = await resolve_hop(pseudo_hop, state, verifier=llm_judge_verifier)
        result = {"mode": mode, "answer": {"text": r["answer"], "status": "ANSWERED" if r["status"] == "OK" else "UNVERIFIABLE",
                "confidence": 0.5, "path_check": None, "evidence": [], "verified_until": None},
                "hop_results": {1: r}, "hops": [pseudo_hop]}
        tracer.emit({"type": "final", "answer": result["answer"]})
        return result

    dec = decompose(corr["corrected_question"], corr["unresolved"], adapters["llm"],
                    schema_relations=schema, types=types, alias_names=_leak_check_alias_names(adapters))

    if dec["mode"] == "SIMPLE" or not dec["hops"]:
        state = _state(adapters, settings, relation_desc)
        pseudo_hop = {"id": 1, "start": dec["corrected_question"], "start_node": None,
                      "relation": "relatedTo", "relation_grounded": None, "direction": "fwd",
                      "expected": {"type": "", "definition": dec["corrected_question"]},
                      "anchor_meta": {"name": None, "disambiguator": None},
                      "sub_query": dec["corrected_question"], "hint_relations": None}
        tracer.emit({"type": "plan_ready", "hops": [pseudo_hop], "mode": "SIMPLE"})
        r = await resolve_hop(pseudo_hop, state, verifier=verifier)
        result = {"mode": "SIMPLE", "answer": {"text": r["answer"],
                "status": "ANSWERED" if r["status"] == "OK" else "UNVERIFIABLE",
                "confidence": 0.5, "path_check": None, "evidence": [], "verified_until": None},
                "hop_results": {1: r}, "hops": [pseudo_hop]}
        tracer.emit({"type": "final", "answer": result["answer"]})
        return result

    tracer.emit({"type": "plan_ready", "hops": dec["hops"], "mode": "FULL"})
    state = _state(adapters, settings, relation_desc)

    async def _rh(hop, st):
        # 그라운딩은 여기서 hop 단위로 한다({n} 참조는 executor가 이미 실제 답 문자열로 치환한
        # 뒤 이 함수를 호출하므로, 그 시점에야 hop["start"]가 리터럴이 되어 앵커를 조회할 수
        # 있다. 실행 전 일괄 그라운딩은 참조 hop의 start_node를 영구히 None으로 만드는 버그였다.)
        g = ground([hop], graph=adapters["graph"], alias=alias, embedder=adapters["embedder"],
                   llm=adapters["llm"], relation_desc=relation_desc,
                   theta_high=settings["grounding"]["theta_high"],
                   theta_low=settings["grounding"]["theta_low"])[0]
        return await resolve_hop(g, st, verifier=verifier)

    hop_results = await execute(dec["hops"], _rh, state)
    answer = finalize(dec["hops"], hop_results, dec.get("final_op"), graph=adapters["graph"], alias=alias)
    tracer.emit({"type": "final", "answer": answer})
    return {"mode": mode, "answer": answer, "hop_results": hop_results, "hops": dec["hops"],
            "decompose": dec, "corrections": corr}


def _state(adapters, settings, relation_desc):
    v, e = settings["verification"], settings["execution"]
    return {"vector": adapters["vector"], "graph": adapters["graph"], "llm": adapters["llm"],
            "embedder": adapters["embedder"], "alias": adapters["alias"], "tracer": adapters["tracer"],
            "pass_ratio": v["pass_ratio"], "theta_desc": v["theta_desc"], "score_lambda": v["score_lambda"],
            "top_k": settings["retrieval"]["top_k_paragraphs"], "max_requery": e["max_requery"],
            "tie_epsilon": e["tie_epsilon"], "relation_desc": relation_desc,
            "use_hyde": settings["retrieval"]["use_hyde_definition"],
            "pid2text": adapters["vector"].pid2text}       # 그래프 근거 문단 desc_check용 (FaissIndex 재사용)


def build_pipeline(mode="FULL", tracer=None):
    """returns callable(question:str) -> awaitable result dict. bootstrap의 유일한 조립 지점.
    tracer 주입은 데모 UI의 실시간 4패널 시각화용(각 arm이 자기 QueueTracer를 가져야 이벤트가
    패널끼리 안 섞인다) — eval 하네스는 기본값(JsonTracer)을 그대로 쓰므로 영향 없음."""
    settings = _load_settings()
    schema, relation_desc, types = _load_relations()
    adapters = build_adapters(settings, tracer=tracer)
    verifier = llm_judge_verifier if mode in ("ours_g", "agent_basic") else None
    run_mode = "FULL" if mode == "ours_g" else mode

    async def pipeline(question):
        return await run_pipeline(question, adapters, settings, schema, relation_desc, types,
                                  mode=run_mode, verifier=verifier)
    return pipeline
