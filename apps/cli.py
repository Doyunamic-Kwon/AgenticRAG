#!/usr/bin/env python3
"""단일 질문 디버깅 진입점. Streamlit 없이 파이프라인 1회 실행 + trace 덤프.
usage: python -m apps.cli "질문" [--mode ours]
"""
import sys, asyncio, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="FULL", choices=["FULL", "SIMPLE", "ours_g", "agent_basic", "baseline"])
    a = ap.parse_args()

    from verihop.bootstrap import build_pipeline
    pipeline = build_pipeline(a.mode)
    result = asyncio.run(pipeline(a.question))

    print(f"\n질문: {a.question}")
    print(f"모드: {a.mode}")
    if "decompose" in result:
        for h in result["hops"]:
            print(f"  hop{h['id']}: {h['start']} --{h['relation']}--> ? "
                  f"(grounded={h.get('relation_grounded')}, node={h.get('start_node')})")
    for hid, r in sorted(result["hop_results"].items()):
        print(f"  hop{hid} 결과: {r['status']} → {r['answer']!r} (재질의 {r['retries']}회, 후보 {len(r['candidates'])}개)")
    ans = result["answer"]
    print(f"\n답: {ans['text']}")
    print(f"상태: {ans['status']} | confidence: {ans['confidence']:.2f} | path_check: {ans['path_check']}")


if __name__ == "__main__":
    main()
