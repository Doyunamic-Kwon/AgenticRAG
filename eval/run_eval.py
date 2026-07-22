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
        rows = [json.loads(l) for l in open(ROOT / "data/eval/multihop.jsonl", encoding="utf-8")]
        return [{"qid": r["qid"], "question": r["question"],
                 "gold": {r["hops"][-1]["evidence_paragraph"]}} for r in rows]
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


def main(mode, which, limit):
    c = yaml.safe_load(open(ROOT / "configs/settings.yaml", encoding="utf-8"))
    evalset = load_eval(which, limit)
    print(f"[{mode}/{which}] {len(evalset)}문항")

    if mode == "baseline":
        embedder = make_embedder(c)
        index, ids = load_index(c)
        rows = run_baseline(evalset, embedder, index, ids)
    else:
        raise SystemExit(f"--mode {mode} 는 파이프라인 완성 후(W4). 지금은 baseline만.")

    agg = metrics.aggregate([{"ranked": r["ranked"], "gold": set(r["gold"])} for r in rows])
    run_id = f"{mode}_{which}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"mode": mode, "set": which, **agg}, open(out / "metrics.json", "w"),
              ensure_ascii=False, indent=2)
    with open(out / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"→ results/{run_id}/metrics.json")
    print(f"  Recall@5={agg['recall@5']:.1%}  @10={agg['recall@10']:.1%}  "
          f"MRR={agg['mrr']:.3f}  nDCG@10={agg['ndcg@10']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="baseline", choices=["baseline", "basic", "ours_g", "ours"])
    ap.add_argument("--set", dest="which", default="multi", choices=["single", "multi", "typo"])
    ap.add_argument("--limit", type=int, default=500)
    a = ap.parse_args()
    main(a.mode, a.which, a.limit)
