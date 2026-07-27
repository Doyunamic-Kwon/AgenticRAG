#!/usr/bin/env python3
"""VeriHop 데모 — 4분할 실시간 시각화 (Docs/design.md 기준).
usage: streamlit run apps/streamlit_app.py

Baseline · Agent-basic · Ours-G · Ours 4개 arm을 질문 하나로 동시 실행하고, 각자의 hop이
resolve되는 과정(계획 → 후보 탐색 → 검증 → 재질의 → 확정)을 실시간으로 보여준다. Ours 패널은
추가로 vis-network 그래프 미니뷰로 앵커→후보 엣지가 검증 결과에 따라 생겨나는 걸 보여준다.
"""
import sys, json, asyncio, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

st.set_page_config(page_title="VeriHop", page_icon="🕸️", layout="wide")

ARMS = [
    {"key": "baseline",    "mode": "baseline",    "label": "Baseline",    "color": "#5B6472"},
    {"key": "agent_basic", "mode": "agent_basic", "label": "Agent-basic", "color": "#E3A008"},
    {"key": "ours_g",      "mode": "ours_g",      "label": "Ours−G", "color": "#8B7FD6"},
    {"key": "ours",        "mode": "FULL",        "label": "Ours",       "color": "#4FD1C5"},
]
SCENARIOS = {
    "A · 기본": "베토벤이 태어난 도시가 속한 나라는?",
    "B · 하이라이트": "모차르트가 태어난 도시의 대표 축제는?",
    "C · 오타": "베토밴 교향곡 9번 초연 장소",
}
CHECK_ICON = {True: "✅", False: "❌", None: "⬜"}
BACKLINK_LABEL = {"pass": "pass", "contradict": "contradict(거부)", "absent": "absent", None: "—"}
BACKLINK_EDGE_COLOR = {"pass": "#3FB950", "contradict": "#F85149", "absent": "#D29922", None: "#8B93A7"}


# ── 무거운 리소스는 세션당 1회만 로드 (arm마다 다시 안 만듦) ────────────────────
@st.cache_resource(show_spinner="파이프라인 로딩 중(FAISS·그래프·alias)...")
def get_base_adapters():
    from verihop.bootstrap import _load_settings, _load_relations, build_adapters
    settings = _load_settings()
    schema, relation_desc, types = _load_relations()
    adapters = build_adapters(settings)          # tracer는 자리표시자(JsonTracer) — arm마다 교체해 씀
    return adapters, settings, schema, relation_desc, types


def _arm_view(base_adapters, tracer):
    """base_adapters의 무거운 리소스(FAISS·그래프·alias·llm·embedder)는 그대로 재사용하되
    tracer만 이 arm 전용으로 바꾼 얕은 사본. 4-arm이 동시에 도는데 tracer를 공유하면 이벤트가
    패널끼리 섞인다."""
    return {**base_adapters, "tracer": tracer}


def _run_baseline(question, adapters):
    """baseline arm: 검색만, 답변 생성도 검증도 없음(05 §1 정의 그대로)."""
    hits = adapters["vector"].search(question, 8)   # [(pid, text, score)]
    return {"hits": [{"pid": pid, "text": text[:200], "score": score} for pid, text, score in hits]}


def _run_pipeline_arm(question, mode, adapters, settings, schema, relation_desc, types):
    from verihop.bootstrap import run_pipeline, llm_judge_verifier
    verifier = llm_judge_verifier if mode in ("ours_g", "agent_basic") else None
    run_mode = "FULL" if mode == "ours_g" else mode
    return asyncio.run(run_pipeline(question, adapters, settings, schema, relation_desc, types,
                                    mode=run_mode, verifier=verifier))


def _worker(arm, question, base_adapters, settings, schema, relation_desc, types, tracer, results):
    try:
        if arm["mode"] == "baseline":
            results[arm["key"]] = {"ok": True, "data": _run_baseline(question, _arm_view(base_adapters, tracer))}
        else:
            r = _run_pipeline_arm(question, arm["mode"], _arm_view(base_adapters, tracer),
                                  settings, schema, relation_desc, types)
            results[arm["key"]] = {"ok": True, "data": r}
    except Exception as e:
        results[arm["key"]] = {"ok": False, "error": str(e)}
    finally:
        tracer.emit({"type": "__worker_done__"})


# ── 이벤트를 패널 상태로 접기 ────────────────────────────────────────────────
def _new_panel_state():
    return {"status": "idle", "plan": None, "hops": {}, "final": None, "graph_edges": [], "error": None}


def _fold_event(state, ev):
    t = ev.get("type")
    if t == "plan_ready":
        state["plan"] = ev["hops"]
        state["status"] = "running"
    elif t == "hop_start":
        state["hops"].setdefault(ev["hop_id"], {"start": ev["start"], "relation": ev["relation"],
                                                 "direction": ev["direction"], "candidates": [],
                                                 "status": "running", "answer": None})
    elif t == "candidate_verified":
        hop = state["hops"].setdefault(ev["hop_id"], {"candidates": [], "status": "running"})
        hop["candidates"].append(ev)
        start = hop.get("start")
        if start:
            state["graph_edges"].append({
                "from": start, "to": ev["entity"],
                "backlink": (ev["verification"] or {}).get("backlink_check"),
                "passed": (ev["verification"] or {}).get("passed"),
            })
    elif t == "hop_resolved":
        hop = state["hops"].setdefault(ev["hop_id"], {"candidates": [], "status": "running"})
        hop["status"], hop["answer"] = ev["status"], ev["answer"]
    elif t == "requery":
        hop = state["hops"].setdefault(ev["hop_id"], {"candidates": [], "status": "running"})
        hop.setdefault("requeries", []).append(ev["new_query"])
    elif t == "final":
        state["final"] = ev["answer"]
        state["status"] = "done"
    elif t == "__worker_done__":
        if state["status"] != "done":
            state["status"] = "done"


# ── 렌더링 ───────────────────────────────────────────────────────────────────
def _graph_html(edges, height=220):
    """vis-network로 앵커→후보 엣지를 backlink 결과 색으로 그린다(Chart.js가 아니라 노드-엣지 전용
    라이브러리 — design.md 결정사항). 매 폴링 tick마다 누적분 전체를 다시 그리는 방식(Streamlit
    components는 rerun마다 iframe이 새로 뜨므로 diff 애니메이션 대신 단계적 누적으로 "생겨나는" 느낌)."""
    nodes, seen = [], set()
    vis_edges = []
    for i, e in enumerate(edges):
        for n in (e["from"], e["to"]):
            if n not in seen:
                seen.add(n)
                nodes.append({"id": n, "label": n,
                              "color": "#4FD1C5" if n == edges[0]["from"] else "#1A1F2B",
                              "font": {"color": "#E4E7EC"}})
        color = BACKLINK_EDGE_COLOR.get(e["backlink"])
        dashes = e["backlink"] == "contradict"
        vis_edges.append({"id": i, "from": e["from"], "to": e["to"], "color": color,
                          "dashes": dashes, "arrows": "to",
                          "label": BACKLINK_LABEL.get(e["backlink"], "")})
    nodes_json, edges_json = json.dumps(nodes, ensure_ascii=False), json.dumps(vis_edges, ensure_ascii=False)
    return f"""
    <div id="graph" style="width:100%;height:{height}px;background:#0B0E14;border-radius:8px;"></div>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script>
      const nodes = new vis.DataSet({nodes_json});
      const edges = new vis.DataSet({edges_json});
      new vis.Network(document.getElementById("graph"), {{nodes, edges}}, {{
        physics: {{stabilization: true}},
        nodes: {{shape: "box", borderWidth: 1, color: {{border: "#262C3A"}}, margin: 8}},
        edges: {{width: 2, smooth: true}},
      }});
    </script>
    """


def _render_hop_card(hop_id, hop):
    badge = {"OK": "🟢", "FAILED": "🔴"}.get(hop.get("status"), "🟡")
    lines = [f"**{badge} hop{hop_id}** — `{hop.get('start','?')}` --**{hop.get('relation','?')}**"
             f"({hop.get('direction','?')})--> **{hop.get('answer') or '...'}**"]
    for rq in hop.get("requeries", []):
        lines.append(f"  ↻ 재질의: _{rq}_")
    for c in sorted(hop.get("candidates", []), key=lambda x: x["score"], reverse=True)[:5]:
        v = c["verification"] or {}
        mark = "✔️" if v.get("passed") else "・"
        lines.append(f"  {mark} `{c['entity']}` ({c['origin']}) "
                     f"type{CHECK_ICON.get(v.get('type_check'))} name{CHECK_ICON.get(v.get('name_check'))} "
                     f"desc{CHECK_ICON.get(v.get('desc_check'))} backlink:{BACKLINK_LABEL.get(v.get('backlink_check'))}")
    return "\n\n".join(lines)


def render_panel(arm, state, placeholder):
    with placeholder.container():
        st.markdown(f'<div style="border-bottom:2px solid {arm["color"]};padding-bottom:8px;'
                   f'font-weight:600;letter-spacing:.02em;text-transform:uppercase;">'
                   f'{arm["label"]}</div>', unsafe_allow_html=True)
        dot = {"idle": "⚪", "running": "🟡", "done": "🟢"}.get(state["status"], "⚪")
        st.caption(f"{dot} {state['status']}")

        if state.get("error"):
            st.error(state["error"])
            return

        if arm["key"] == "baseline":
            data = state.get("_baseline")
            if data:
                for h in data["hits"][:5]:
                    st.markdown(f"`{h['pid']}` score={h['score']:.2f}  \n{h['text']}")
            return

        if state["plan"]:
            plan_txt = " → ".join(f"hop{h['id']}:{h['relation']}" for h in state["plan"]
                                  if h.get("relation") != "relatedTo") or "(단순 검색 — SIMPLE)"
            st.caption(f"계획: {plan_txt}")

        for hop_id, hop in sorted(state["hops"].items()):
            st.markdown(_render_hop_card(hop_id, hop))

        if arm["key"] == "ours" and state["graph_edges"]:
            import streamlit.components.v1 as components
            components.html(_graph_html(state["graph_edges"]), height=230)

        if state["final"]:
            f = state["final"]
            color = {"ANSWERED": "green", "PARTIAL": "orange", "UNVERIFIABLE": "orange"}.get(f["status"], "gray")
            st.markdown(f":{color}[**{f['status']}**] 확신도 {f['confidence']:.0%}")
            st.markdown(f"### {f['text'] or '(답 없음)'}")


# ── UI ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { background: #0B0E14; }
</style>
""", unsafe_allow_html=True)

st.title("🕸️ VeriHop")
st.caption("한국어 복합 질문을 hop 단위로 분해하고, 지식그래프 4중 검증(type·name·desc·backlink)으로 "
          "틀린 중간 결과를 답이 되기 전에 걸러낸다. 4개 arm을 동시 실행해 비교한다.")

preset = st.radio("데모 시나리오", list(SCENARIOS.keys()) + ["직접 입력"], horizontal=True)
question = st.text_input("질문", SCENARIOS.get(preset, ""), label_visibility="collapsed") \
    if preset == "직접 입력" else SCENARIOS[preset]
go = st.button("4-arm 동시 실행", type="primary")

if "panel_states" not in st.session_state:
    st.session_state.panel_states = {a["key"]: _new_panel_state() for a in ARMS}

cols = st.columns(4)
placeholders = {a["key"]: cols[i].empty() for i, a in enumerate(ARMS)}

if go and question.strip():
    base_adapters, settings, schema, relation_desc, types = get_base_adapters()
    tracers, results = {}, {}
    st.session_state.panel_states = {a["key"]: _new_panel_state() for a in ARMS}
    threads = []
    for arm in ARMS:
        from verihop.adapters.queue_tracer import QueueTracer
        qt = QueueTracer()
        tracers[arm["key"]] = qt
        th = threading.Thread(target=_worker, args=(arm, question, base_adapters, settings,
                                                     schema, relation_desc, types, qt, results), daemon=True)
        threads.append(th)
        th.start()

    t0 = time.time()
    while any(th.is_alive() for th in threads) and time.time() - t0 < 600:
        for arm in ARMS:
            state = st.session_state.panel_states[arm["key"]]
            for ev in tracers[arm["key"]].drain():
                _fold_event(state, ev)
            if arm["key"] in results and not results[arm["key"]]["ok"]:
                state["error"] = results[arm["key"]]["error"]
            if arm["key"] == "baseline" and arm["key"] in results and results[arm["key"]]["ok"]:
                state["_baseline"] = results[arm["key"]]["data"]
                state["status"] = "done"
            render_panel(arm, state, placeholders[arm["key"]])
        time.sleep(0.4)

    for arm in ARMS:                                   # 마지막 상태 한 번 더 반영
        state = st.session_state.panel_states[arm["key"]]
        for ev in tracers[arm["key"]].drain():
            _fold_event(state, ev)
        if arm["key"] in results:
            if not results[arm["key"]]["ok"]:
                state["error"] = results[arm["key"]]["error"]
            elif arm["key"] == "baseline":
                state["_baseline"] = results[arm["key"]]["data"]
        state["status"] = "done"
        render_panel(arm, state, placeholders[arm["key"]])
else:
    for arm in ARMS:
        render_panel(arm, st.session_state.panel_states[arm["key"]], placeholders[arm["key"]])
    if not go:
        st.info("시나리오를 고르거나 질문을 입력하고 '4-arm 동시 실행'을 누르세요.")
