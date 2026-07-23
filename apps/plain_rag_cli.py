#!/usr/bin/env python3
"""일반 RAG 대조군 데모 진입점 (분해·그래프·검증 없음, 임베딩 검색+LLM 답변만).
usage: python -m apps.plain_rag_cli "질문"
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from verihop.plain_rag import answer
    question = " ".join(sys.argv[1:]) or input("질문: ")
    r = answer(question)
    print(f"\n질문: {question}")
    print(f"답변: {r['text']}")
    print(f"근거 문단: {r['evidence_paragraph_ids']}")


if __name__ == "__main__":
    main()
