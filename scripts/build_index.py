#!/usr/bin/env python3
"""W1.2 벡터 인덱스 구축 (FAISS flat, cosine) + 검색 스팟체크.

우리 규모(≤3만 문단)에선 exact 검색이 즉시라 IndexFlatIP(정규화=코사인) flat이면 충분.
usage:
  python3 scripts/build_index.py --selftest    # API 없이 FAISS 파이프라인 검증(랜덤 벡터)
  python3 scripts/build_index.py                # 실제 인덱싱 (.env 의 임베딩 키 필요)
  python3 scripts/build_index.py --spotcheck 20 # 인덱싱 후 dev 20문항 Recall@10 목측
out: data/index/{faiss.index, ids.jsonl, vectors.npy}
"""
import sys, os, json, argparse, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import faiss
import yaml


def cfg():
    return yaml.safe_load(open(ROOT / "configs/settings.yaml", encoding="utf-8"))


def build_faiss(vecs, ids, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    v = np.ascontiguousarray(vecs, dtype="float32")
    faiss.normalize_L2(v)                       # 정규화 후 내적 = 코사인
    index = faiss.IndexFlatIP(v.shape[1])
    index.add(v)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "ids.jsonl", "w", encoding="utf-8") as f:
        for r in ids:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(out_dir / "vectors.npy", v)
    return index


def make_embedder():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    c = cfg()
    prov = c["providers"]["embedding"]
    ec = c["embedding"][prov]
    key_env = {"upstage": "UPSTAGE_API_KEY", "openai": "OPENAI_API_KEY",
               "naver": "NAVER_CLOVA_API_KEY"}[prov]
    api_key = os.environ.get(key_env)
    assert api_key, f"{key_env} 없음 — .env 에 넣어주세요 (.env.example 참고)"
    assert "base_url" in ec, f"{prov} 는 OpenAI 호환이 아님 — 전용 어댑터 필요"
    from verihop.adapters.openai_embedder import OpenAIEmbedder
    return OpenAIEmbedder(api_key, ec["base_url"], ec["model"], ec["dim"]), prov, ec


def load_corpus():
    corpus = [json.loads(l) for l in open(ROOT / "data/corpus.jsonl", encoding="utf-8")]
    texts = [c["text"] for c in corpus]
    ids = [{"paragraph_id": c["paragraph_id"], "doc_title": c["doc_title"]} for c in corpus]
    return corpus, texts, ids


def main(spotcheck_n):
    embedder, prov, ec = make_embedder()
    out = ROOT / cfg()["vector"]["dir"]
    corpus, texts, ids = load_corpus()
    print(f"임베딩 {len(texts)}문단 ({prov}/{ec['model']}, dim {ec['dim']})...")

    cache = out / "vectors.npy"
    if cache.exists() and np.load(cache).shape[0] == len(texts):
        print("캐시된 임베딩 재사용 (vectors.npy). 제공자/모델 바꿨으면 data/index 삭제 후 재실행")
        vecs = np.load(cache)
    else:
        vecs = np.asarray(embedder.embed(texts), dtype="float32")
    index = build_faiss(vecs, ids, out)
    print(f"인덱스 저장 → {out}  ({index.ntotal} vec, dim {index.d})")

    if spotcheck_n:
        spotcheck(embedder, index, ids, corpus, spotcheck_n)


def spotcheck(embedder, index, ids, corpus, n):
    """dev 질문 n개로 Recall@10 목측. gold = 질문 문맥이 속한 corpus 문단."""
    text2pid = {c["text"]: c["paragraph_id"] for c in corpus}
    dev = json.load(open(ROOT / "data/raw/KorQuAD_v1.0_dev.json", encoding="utf-8"))["data"]
    samples = []
    for doc in dev:
        for para in doc["paragraphs"]:
            pid = text2pid.get(para["context"].strip())
            if pid:
                for qa in para["qas"]:
                    samples.append((qa["question"], pid))
    random.seed(0)
    samples = random.sample(samples, min(n, len(samples)))
    qvecs = np.asarray(embedder.embed([q for q, _ in samples], is_query=True), dtype="float32")
    faiss.normalize_L2(qvecs)
    _, I = index.search(qvecs, 10)
    hit = sum(1 for row, (_, gold) in zip(I, samples)
              if gold in {ids[j]["paragraph_id"] for j in row})
    print(f"[스팟체크] Recall@10 = {hit}/{len(samples)} = {hit/len(samples):.1%} "
          f"({'통과' if hit/len(samples) >= 0.7 else '미달 → 한국어 특화 임베딩 교체 검토'})")


def selftest():
    ids = [{"paragraph_id": f"p{i}", "doc_title": "t"} for i in range(100)]
    v = np.random.rand(100, 32).astype("float32")
    out = ROOT / "data/index_selftest"
    index = build_faiss(v, ids, out)
    q = v[:1].copy(); faiss.normalize_L2(q)
    _, I = index.search(q, 5)
    assert I[0][0] == 0, "self-query가 자기 자신을 top1로 못 찾음"
    import shutil; shutil.rmtree(out)
    print(f"selftest OK: FAISS build+query 정상 (top5={I[0].tolist()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--spotcheck", type=int, default=0)
    a = ap.parse_args()
    selftest() if a.selftest else main(a.spotcheck)
