"""TracerPort 구현. JSON 구조화 로그 (request_id 관통, 03 [로깅]).
implements: verihop.ports.TracerPort
메모리에 이벤트를 모으고 dump()로 파일/UI에 넘긴다.
서드파티 import 없음(stdlib json). usecase는 이 파일의 존재를 모른다.
"""
from __future__ import annotations
import json


class JsonTracer:
    def __init__(self):
        self.events: list[dict] = []

    def emit(self, event):
        self.events.append(event)

    def dump(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
