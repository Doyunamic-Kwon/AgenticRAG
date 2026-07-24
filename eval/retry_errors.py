#!/usr/bin/env python3
"""완료된 평가 run의 ERROR 상태 문항만 재시도해 그 자리에서 패치 + metrics 재집계.
인터넷 단절 등으로 일부만 실패했을 때 127개 전체를 다시 돌리지 않기 위함.

usage: python3 eval/retry_errors.py results/{run_id}
"""
import sys, os, json, argparse, asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import _norm, EVAL_WORKERS


def main(run_dir):
    d = ROOT / run_dir
    rows = [json.loads(l) for l in open(d / "per_question.jsonl", encoding="utf-8")]
    meta = json.load(open(d / "metrics.json", encoding="utf-8"))
    mode = meta["mode"]
    errs = [r for r in rows if r["status"] == "ERROR"]
    print(f"[{run_dir}] 총 {len(rows)} · ERROR {len(errs)}건 재시도")
    if not errs:
        print("재시도할 ERROR 없음."); return

    mh = ROOT / "data/eval/multihop_2wiki.jsonl"
    gold_answers = {}
    for l in open(mh, encoding="utf-8"):
        g = json.loads(l)
        gold_answers[g["qid"]] = {g["answer"]} | set(g.get("answer_aliases", []))

    from verihop.bootstrap import build_pipeline
    pipeline = build_pipeline(mode)

    def _retry_one(e):
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
        print(f"  재시도 {row['qid']}: {row['status']} em={row['em']} → {row['answer']}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
        patched = list(ex.map(_retry_one, errs))
    patched_by_qid = {r["qid"]: r for r in patched}
    rows = [patched_by_qid.get(r["qid"], r) for r in rows]

    still_err = sum(1 for r in rows if r["status"] == "ERROR")
    n = len(rows) or 1
    em_scored = [r for r in rows if r["em"] is not None]
    agg = {
        "em_accuracy": sum(1 for r in em_scored if r["em"]) / (len(em_scored) or 1),
        "em_scored_n": len(em_scored),
        "avg_confidence": sum(r["confidence"] for r in rows) / n,
        "status_counts": {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}},
        "note": meta.get("note", ""),
    }
    json.dump({"mode": mode, "set": meta["set"], **agg}, open(d / "metrics.json", "w"),
              ensure_ascii=False, indent=2)
    with open(d / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n패치 완료 → {run_dir}/metrics.json | 남은 ERROR {still_err}")
    print(f"  EM정답률={agg['em_accuracy']:.1%} (n={agg['em_scored_n']})  status={agg['status_counts']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    main(a.run_dir)
