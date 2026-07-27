"""검색 품질 지표 — 순수 함수 (05 §3). ranked=문단ID 리스트(상위순), gold=정답 문단ID 집합.
gold가 문항당 1개뿐이라 hits_at_k(top-k 안에 있으면 1, 없으면 0)가 전통적 Recall@k(여러
정답 중 몇 개를 찾았는지 비율)와 수학적으로 동일해진다 — 그래서 지표명은 Recall이 아니라
Hits@k(=Hit Rate@k)로 정확히 부른다(2026-07-27)."""
import math


def hits_at_k(ranked, gold, k):
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
        "hits@5": sum(hits_at_k(r["ranked"], r["gold"], 5) for r in rows) / n,
        "hits@10": sum(hits_at_k(r["ranked"], r["gold"], 10) for r in rows) / n,
        "mrr": sum(mrr(r["ranked"], r["gold"]) for r in rows) / n,
        "ndcg@10": sum(ndcg_at_k(r["ranked"], r["gold"], 10) for r in rows) / n,
    }


def demo():
    assert hits_at_k(["a", "b", "c"], {"c"}, 5) == 1.0
    assert hits_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
    assert mrr(["a", "b", "c"], {"b"}) == 0.5
    assert abs(ndcg_at_k(["a", "b"], {"a"}, 10) - 1.0) < 1e-9
    assert abs(ndcg_at_k(["b", "a"], {"a"}, 10) - 1 / math.log2(3)) < 1e-9
    print("metrics demo OK")


if __name__ == "__main__":
    demo()
