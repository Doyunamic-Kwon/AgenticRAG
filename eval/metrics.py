"""검색 품질 지표 — 순수 함수 (05 §3). ranked=문단ID 리스트(상위순), gold=정답 문단ID 집합."""
import math


def recall_at_k(ranked, gold, k):
    return 1.0 if any(p in gold for p in ranked[:k]) else 0.0


def mrr(ranked, gold):
    for i, p in enumerate(ranked, 1):
        if p in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, gold, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(ranked[:k], 1) if p in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def aggregate(rows):
    """rows: [{ranked, gold}] → 평균 지표 dict."""
    n = len(rows) or 1
    return {
        "n": len(rows),
        "recall@5": sum(recall_at_k(r["ranked"], r["gold"], 5) for r in rows) / n,
        "recall@10": sum(recall_at_k(r["ranked"], r["gold"], 10) for r in rows) / n,
        "mrr": sum(mrr(r["ranked"], r["gold"]) for r in rows) / n,
        "ndcg@10": sum(ndcg_at_k(r["ranked"], r["gold"], 10) for r in rows) / n,
    }


def demo():
    assert recall_at_k(["a", "b", "c"], {"c"}, 5) == 1.0
    assert recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
    assert mrr(["a", "b", "c"], {"b"}) == 0.5
    assert abs(ndcg_at_k(["a", "b"], {"a"}, 10) - 1.0) < 1e-9
    assert abs(ndcg_at_k(["b", "a"], {"a"}, 10) - 1 / math.log2(3)) < 1e-9
    print("metrics demo OK")


if __name__ == "__main__":
    demo()
