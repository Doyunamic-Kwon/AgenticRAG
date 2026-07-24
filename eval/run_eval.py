#!/usr/bin/env python3
"""평가 하네스 (05_평가_명세). W1.5는 baseline 모드. basic/ours_g/ours는 파이프라인 완성 후(W4).

usage: python3 eval/run_eval.py --mode baseline --set multi [--limit 500]
out: results/{run_id}/metrics.json + per_question.jsonl
top-k 랭킹 리스트: baseline은 원질의 1회 검색 결과 상위 k (05 §3 / C3).
"""
import sys, os, json, argparse, random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))     # metrics
import numpy as np
import faiss
import yaml
from dotenv import load_dotenv
import metrics
from verihop.usecases.correct import _strip_particle   # EM 채점에 조사 정규화 재사용 (05 "형태소 정규화")

KEY_ENV = {"upstage": "UPSTAGE_API_KEY", "openai": "OPENAI_API_KEY", "naver": "NAVER_CLOVA_API_KEY"}


def make_embedder(c):
    load_dotenv(ROOT / ".env")
    p = c["providers"]["embedding"]; ec = c["embedding"][p]
    from verihop.adapters.openai_embedder import OpenAIEmbedder
    return OpenAIEmbedder(os.environ[KEY_ENV[p]], ec["base_url"], ec["model"], ec["dim"],
                          batch=c["embedding"].get("batch", 32), max_chars=c["embedding"].get("max_chars", 5000))


def load_index(c):
    d = ROOT / c["vector"]["dir"]
    index = faiss.read_index(str(d / "faiss.index"))
    ids = [json.loads(l)["paragraph_id"] for l in open(d / "ids.jsonl", encoding="utf-8")]
    return index, ids


def load_eval(which, limit):
    """returns [{qid, question, gold:set(pid)}]."""
    if which == "multi":
        # 2Wiki 적응 gold (깨끗한 셋). KorQuAD 브릿지 초판(multihop.jsonl)은 품질 미달로 미사용.
        rows = [json.loads(l) for l in open(ROOT / "data/eval/multihop_2wiki.jsonl", encoding="utf-8")]
        out = [{"qid": r["qid"], "question": r["question"],
                "gold": {r["hops"][-1]["evidence_paragraph"]}} for r in rows]
        return out[:limit]
    if which == "single":
        text2pid = {json.loads(l)["text"]: json.loads(l)["paragraph_id"]
                    for l in open(ROOT / "data/corpus.jsonl", encoding="utf-8")}
        dev = json.load(open(ROOT / "data/raw/KorQuAD_v1.0_dev.json", encoding="utf-8"))["data"]
        out = []
        for doc in dev:
            for para in doc["paragraphs"]:
                pid = text2pid.get(para["context"].strip())
                if pid:
                    for qa in para["qas"]:
                        out.append({"qid": qa["id"], "question": qa["question"], "gold": {pid}})
        random.seed(0)
        return random.sample(out, min(limit, len(out)))
    raise SystemExit(f"--set {which} 미지원 (typo 셋은 W4.2에서 생성)")


def run_baseline(evalset, embedder, index, ids):
    """원질의 1회 벡터검색 → 상위 10 랭킹 (05 §2 Baseline)."""
    qv = np.asarray(embedder.embed([e["question"] for e in evalset], is_query=True), dtype="float32")
    faiss.normalize_L2(qv)
    _, I = index.search(qv, 10)
    rows = []
    for e, row in zip(evalset, I):
        ranked = [ids[j] for j in row]
        rows.append({"qid": e["qid"], "ranked": ranked, "gold": list(e["gold"])})
    return rows


def load_eval_answers(which):
    """qid → {answer, aliases} (파이프라인 모드의 EM 채점용, multi 셋만)."""
    if which != "multi":
        return {}
    out = {}
    for l in open(ROOT / "data/eval/multihop_2wiki.jsonl", encoding="utf-8"):
        r = json.loads(l)
        out[r["qid"]] = {r["answer"]} | set(r.get("answer_aliases", []))
    return out


def _norm(s):
    s = "".join((s or "").split())
    s = _strip_particle(s)          # 답 끝에 붙은 조사(은/는/이/가/을/를/...) 제거 후 비교 (05 형태소 정규화)
    return s.lower()


EVAL_WORKERS = 2  # 문제: 3워커+max_retries 6에서도 ours_g 120건 중 112건이 그대로 429 —
# 단일 격리 호출은 즉시 성공(계정 쿼터 소진 아님, 순간 동시부하형 RPM 리밋). 3스레드 동시 재시도가
# 서로 백오프 타이밍이 겹쳐(thundering herd) 같은 포화 구간에 반복 진입한 것으로 추정. 2로 인하.
# 이마저 안 되면 1(완전 순차)로 내릴 것 — 더 이상의 동시성 튜닝 시도는 여기서 중단.


async def run_pipeline_mode(evalset, gold_answers, mode):
    """ours/ours_g/agent_basic: 실제 파이프라인 실행 → EM 기반 정답률(05 C3 Recall@k 전체는 후속,
    resolve_hop이 검색시점 문단 노출해야 함 — 오늘은 파이프라인 회복 자체를 EM으로 증명).
    문제: llm.complete()/embedder.embed()가 usecase 내부에서 동기 블로킹 호출이라(await로 안 감쌈)
    async def 구조여도 문항 간 병렬성이 전혀 없었다 — 127문항 순차 실행에 문항당 ~3분(hop별 LLM
    왕복 다수)씩 걸려 EVAL_WORKERS개 스레드로 문항을 동시 처리(각 스레드가 자체 이벤트루프로
    asyncio.run 실행 — build_graph.py의 ThreadPoolExecutor 패턴과 동일)."""
    sys.path.insert(0, str(ROOT / "src"))
    from concurrent.futures import ThreadPoolExecutor
    from verihop.bootstrap import build_pipeline
    import asyncio
    pipeline = build_pipeline(mode)

    def _run_one(e):
        try:
            r = asyncio.run(pipeline(e["question"]))
            ans = r["answer"]
            gold = gold_answers.get(e["qid"], set())
            em = any(_norm(ans["text"]) == _norm(g) or _norm(g) in _norm(ans["text"]) for g in gold) if gold else None
            row = {"qid": e["qid"], "question": e["question"], "answer": ans["text"],
                   "status": ans["status"], "confidence": ans["confidence"], "em": em,
                   "gold": list(gold)}
        except Exception as ex:
            row = {"qid": e["qid"], "question": e["question"], "answer": None,
                   "status": "ERROR", "confidence": 0.0, "em": False, "error": str(ex)}
        print(f"  {row['qid']}: {row['status']} em={row['em']} → {row['answer']}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
        rows = list(ex.map(_run_one, evalset))
    return rows


def main(mode, which, limit):
    c = yaml.safe_load(open(ROOT / "configs/settings.yaml", encoding="utf-8"))
    evalset = load_eval(which, limit)
    print(f"[{mode}/{which}] {len(evalset)}문항")

    if mode == "baseline":
        embedder = make_embedder(c)
        index, ids = load_index(c)
        rows = run_baseline(evalset, embedder, index, ids)
        agg = metrics.aggregate([{"ranked": r["ranked"], "gold": set(r["gold"])} for r in rows])
    else:
        import asyncio
        gold_answers = load_eval_answers(which)
        rows = asyncio.run(run_pipeline_mode(evalset, gold_answers, mode))
        n = len(rows) or 1
        em_scored = [r for r in rows if r["em"] is not None]
        agg = {
            "em_accuracy": sum(1 for r in em_scored if r["em"]) / (len(em_scored) or 1),
            "em_scored_n": len(em_scored),
            "avg_confidence": sum(r["confidence"] for r in rows) / n,
            "status_counts": {s: sum(1 for r in rows if r["status"] == s)
                              for s in {r["status"] for r in rows}},
            "note": "05 C3 전체 Recall@k(검색시점 합집합 재랭킹)는 후속 — 오늘은 최종답 EM으로 회복 증명",
        }

    run_id = f"{mode}_{which}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"mode": mode, "set": which, **agg}, open(out / "metrics.json", "w"),
              ensure_ascii=False, indent=2)
    with open(out / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"→ results/{run_id}/metrics.json")
    if mode == "baseline":
        print(f"  Recall@5={agg['recall@5']:.1%}  @10={agg['recall@10']:.1%}  "
              f"MRR={agg['mrr']:.3f}  nDCG@10={agg['ndcg@10']:.3f}")
    else:
        print(f"  EM정답률={agg['em_accuracy']:.1%} (n={agg['em_scored_n']})  "
              f"평균confidence={agg['avg_confidence']:.2f}  status={agg['status_counts']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="baseline", choices=["baseline", "agent_basic", "ours_g", "ours"])
    ap.add_argument("--set", dest="which", default="multi", choices=["single", "multi", "typo"])
    ap.add_argument("--limit", type=int, default=500)
    a = ap.parse_args()
    main(a.mode, a.which, a.limit)
