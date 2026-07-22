"""ready-set 루프 (재귀 함수 금지, ADR-3). 의존성 해소된 hop부터 병렬 실행.
reads:  hops[], resolve_hop(주입)
writes: hop_results {hop_id: HopResult}
may-import: verihop.models, re, asyncio (stdlib)
tie 분기(활성 체인 ≤3)는 후속(W3): 지금은 top-1 직렬 해소.
test: tests/usecase/test_executor.py (fake resolve_hop 주입)
done-check: tools/check_layers.sh 통과
"""
from __future__ import annotations
import re
import asyncio

_REF = re.compile(r"\{(\d+)\}")


def _refs(hop):
    return {int(m) for m in _REF.findall(hop["start"])}


def _substitute(hop, done):
    """start의 {n}을 hop n의 답으로 치환한 사본 반환 (실행 시점, ADR-3)."""
    start = hop["start"]
    for n in _refs(hop):
        start = start.replace("{%d}" % n, done[n]["answer"])
    return {**hop, "start": start}


async def execute(hops, resolve_hop, state=None):
    """resolve_hop: async (hop, state) -> HopResult. returns {hop_id: HopResult}."""
    done: dict[int, dict] = {}
    while len(done) < len(hops):
        ready = [h for h in hops if h["id"] not in done and _refs(h) <= done.keys()]
        if not ready:                                 # 미해결 참조로 교착 → 중단
            break
        results = await asyncio.gather(
            *[resolve_hop(_substitute(h, done), state) for h in ready])
        for h, r in zip(ready, results):
            done[h["id"]] = r
    return done


def demo():
    calls = []

    async def fake_resolve(hop, state):
        calls.append((hop["id"], hop["start"]))       # 치환된 start 기록
        return {"hop_id": hop["id"], "answer": f"ans{hop['id']}", "status": "OK"}

    hops = [
        {"id": 1, "start": "베토벤"},
        {"id": 2, "start": "{1}"},                    # hop1 답 참조
        {"id": 3, "start": "{2}"},
    ]
    done = asyncio.run(execute(hops, fake_resolve))
    assert set(done) == {1, 2, 3}
    # 치환 확인: hop2의 start가 hop1 답(ans1)으로 바뀌어 들어감
    starts = dict(calls)
    assert starts[1] == "베토벤" and starts[2] == "ans1" and starts[3] == "ans2", starts

    # 팬아웃: 2,3이 1 참조 → 1 먼저, 2·3 병렬
    fan = [{"id": 1, "start": "x"}, {"id": 2, "start": "{1}"}, {"id": 3, "start": "{1}"}]
    calls.clear()
    d = asyncio.run(execute(fan, fake_resolve))
    assert set(d) == {1, 2, 3} and calls[0][0] == 1
    print("executor demo OK")


if __name__ == "__main__":
    demo()
