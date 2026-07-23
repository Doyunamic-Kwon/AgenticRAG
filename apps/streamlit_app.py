#!/usr/bin/env python3
"""VeriHop 데모 (R6: 계획·hop trace·검증뱃지·재작성질의·반복횟수·근거·출처 + 일반 RAG 비교).
usage: streamlit run apps/streamlit_app.py
"""
import sys, asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

st.set_page_config(page_title="VeriHop", page_icon="🕸️", layout="wide")

SCENARIOS = {
    "A · 기본": "베토벤이 태어난 도시가 속한 나라는?",
    "B · 하이라이트": "모차르트가 태어난 도시의 대표 축제는?",
    "C · 오타": "베토밴 교향곡 9번 초연 장소",
}

STATUS_COLOR = {"ANSWERED": "green", "PARTIAL": "orange", "UNVERIFIABLE": "orange", "ERROR": "red"}
CHECK_ICON = {True: "✅", False: "❌", None: "⬜"}
BACKLINK_ICON = {"pass": "✅ pass", "contradict": "🚫 contradict(거부)", "absent": "⬜ absent", None: "⬜"}


@st.cache_resource(show_spinner="파이프라인 로딩 중(FAISS·그래프·alias)...")
def get_pipeline(mode):
    from verihop.bootstrap import build_pipeline
    return build_pipeline(mode)


@st.cache_resource(show_spinner=False)
def get_plain_rag_adapters():
    from verihop.bootstrap import build_adapters, _load_settings
    s = _load_settings()
    return build_adapters(s), s


def run_ours(question, mode):
    pipeline = get_pipeline(mode)
    return asyncio.run(pipeline(question))


def run_plain_rag(question):
    from verihop.plain_rag import answer
    adapters, settings = get_plain_rag_adapters()
    return answer(question, adapters=adapters, settings=settings)


def render_verification(v):
    if not v:
        return ""
    parts = [
        f"type {CHECK_ICON.get(v.get('type_check'))}",
        f"name {CHECK_ICON.get(v.get('name_check'))}",
        f"desc {CHECK_ICON.get(v.get('desc_check'))}",
        f"backlink {BACKLINK_ICON.get(v.get('backlink_check'))}",
    ]
    return " · ".join(parts)


def render_ours(result):
    hops = result.get("hops", [])
    hop_results = result.get("hop_results", {})
    ans = result["answer"]

    if hops and hops[0].get("relation") != "relatedTo":
        st.markdown("**세운 계획**")
        for h in sorted(hops, key=lambda x: x["id"]):
            st.markdown(f"- hop{h['id']}: `{h['start']}` --**{h['relation']}**({h['direction']})--> "
                       f"`{h['expected']['type'] or '?'}`  \n  _{h['expected']['definition']}_")
    else:
        st.markdown("**세운 계획**: 단순 질문 → SIMPLE 모드(단일 검색)")

    st.markdown("**Hop별 진행 (근거·검증·재질의)**")
    for hid, hr in sorted(hop_results.items()):
        badge = "🟢" if hr["status"] == "OK" else ("🔴" if hr["status"] == "FAILED" else "🟡")
        with st.expander(f"{badge} hop{hid} — {hr['status']} → **{hr['answer'] or '(없음)'}** "
                        f"(재질의 {hr['retries']}회, 후보 {len(hr['candidates'])}개)"):
            if len(hr["queries_used"]) > 1:
                st.caption("재작성한 질의: " + " → ".join(hr["queries_used"]))
            top = sorted(hr["candidates"], key=lambda c: c["score"], reverse=True)[:6]
            for c in top:
                mark = "✔️" if c["verification"] and c["verification"]["passed"] else "・"
                st.markdown(f"{mark} **{c['entity']}** ({c['origin']}, score={c['score']:.2f})  \n"
                          f"　{render_verification(c['verification'])}")

    st.markdown("**최종 응답**")
    color = STATUS_COLOR.get(ans["status"], "gray")
    st.markdown(f":{color}[**{ans['status']}**]  ·  확신도 {ans['confidence']:.0%}  ·  "
              f"path_check: {ans['path_check']}")
    st.markdown(f"### {ans['text'] or '(답 없음)'}")
    if ans["evidence"]:
        st.caption("출처: " + ", ".join(sorted({e["paragraph_id"] for e in ans["evidence"]})))


def render_plain(r):
    st.markdown("**답변**")
    st.markdown(f"### {r['text'] or '(답 없음)'}")
    st.caption("출처: " + ", ".join(r["evidence_paragraph_ids"]))
    st.caption("계획·검증·재질의 없음 — 검색 1회 + LLM 답변 1콜")


# ── UI ────────────────────────────────────────────────────────────────────
st.title("🕸️ VeriHop")
st.caption("한국어 복합 질문을 hop 단위로 분해하고, 지식그래프 4중 검증(type·name·desc·backlink)으로 "
          "틀린 중간 결과를 답이 되기 전에 걸러낸다.")

with st.sidebar:
    st.subheader("질문")
    preset = st.radio("데모 시나리오", list(SCENARIOS.keys()) + ["직접 입력"])
    question = (st.text_input("질문 입력", "") if preset == "직접 입력" else SCENARIOS[preset])
    st.text_area("실행할 질문", question, disabled=True, height=68)
    mode = st.selectbox("VeriHop 모드", ["FULL", "ours_g", "agent_basic"],
                        format_func=lambda m: {"FULL": "Ours (그래프 quad 검증)",
                                               "ours_g": "Ours−G (분해만, llm_judge)",
                                               "agent_basic": "Agent-basic (분해 없음)"}[m])
    show_plain = st.checkbox("일반 RAG와 나란히 비교", value=True)
    go = st.button("실행", type="primary", use_container_width=True)

if go and question.strip():
    cols = st.columns(2) if show_plain else [st.container()]
    with cols[0]:
        st.subheader("VeriHop")
        with st.spinner("파이프라인 실행 중..."):
            try:
                result = run_ours(question, mode)
                render_ours(result)
            except Exception as e:
                st.error(f"실행 실패: {e}")
    if show_plain:
        with cols[1]:
            st.subheader("일반 RAG (대조군)")
            with st.spinner("검색+답변 중..."):
                try:
                    render_plain(run_plain_rag(question))
                except Exception as e:
                    st.error(f"실행 실패: {e}")
elif go:
    st.warning("질문을 입력하세요.")
else:
    st.info("왼쪽에서 데모 시나리오를 고르거나 질문을 입력하고 실행을 누르세요.")
