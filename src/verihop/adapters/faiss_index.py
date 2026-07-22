"""VectorIndexPort 구현. FAISS flat (ADR-11로 Chroma→FAISS). 경로A 문단 검색.
implements: verihop.ports.VectorIndexPort
질의를 embedder로 임베딩(is_query) → 정규화 → 내적(코사인) top_k.
서드파티 import는 이 계층에서만 허용. usecase는 이 파일의 존재를 모른다.
test: usecase 테스트에서 tests/fakes.py로 대체됨
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import faiss


class FaissIndex:
    def __init__(self, index_dir, embedder, corpus_files):
        d = Path(index_dir)
        self.index = faiss.read_index(str(d / "faiss.index"))
        self.ids = [json.loads(l)["paragraph_id"] for l in open(d / "ids.jsonl", encoding="utf-8")]
        self.embedder = embedder
        self.pid2text: dict[str, str] = {}
        for fp in corpus_files:
            p = Path(fp)
            if p.exists():
                for l in open(p, encoding="utf-8"):
                    o = json.loads(l)
                    self.pid2text[o["paragraph_id"]] = o["text"]

    def search(self, query, top_k):
        qv = np.asarray(self.embedder.embed([query], is_query=True), dtype="float32")
        faiss.normalize_L2(qv)
        scores, idx = self.index.search(qv, top_k)
        out: list[tuple[str, str, float]] = []
        for j, s in zip(idx[0], scores[0]):
            if j < 0:
                continue
            pid = self.ids[j]
            out.append((pid, self.pid2text.get(pid, ""), float(s)))
        return out
