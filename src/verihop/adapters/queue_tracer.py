"""TracerPort 구현. 이벤트를 즉시 thread-safe 큐로 흘려보낸다.
implements: verihop.ports.TracerPort
용도: 데모 UI의 실시간 4패널 시각화 — 파이프라인은 별도 스레드에서 돌고, Streamlit 메인 스레드가
이 큐를 폴링하며 화면을 갱신한다(JsonTracer처럼 메모리에 쌓아뒀다 끝나고 dump하는 방식으론
"진행 중"을 못 보여준다). 큐잉만 하고 그리기는 UI 쪽 책임 — usecase는 이 파일의 존재를 모른다.
"""
from __future__ import annotations
import queue


class QueueTracer:
    def __init__(self):
        self.q: queue.Queue = queue.Queue()

    def emit(self, event):
        self.q.put(event)

    def drain(self):
        """지금까지 쌓인 이벤트를 전부 꺼내 리스트로 반환(폴링 루프에서 매 tick 호출)."""
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out
