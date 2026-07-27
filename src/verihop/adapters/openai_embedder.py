"""EmbedderPort 구현 (OpenAI 호환 API). Upstage solar / OpenAI 공용.
implements: verihop.ports.EmbedderPort
solar-embedding은 문서=-passage, 질의=-query 변형을 쓰므로 is_query로 분기한다.
서드파티 import는 이 계층에서만 허용. usecase는 이 파일의 존재를 모른다.
test: usecase 테스트에서 tests/fakes.py로 대체됨

레이트리밋(2026-07-24): openai_llm.py와 같은 계정 쿼터(실측 100 RPM)를 공유하는데 여기는
스로틀이 전혀 없었다 — 특히 ground.py의 앵커 폴백이 배치(32개씩) 반복 호출을 걸어 폴백
1회에 ~200개 요청이 순간적으로 나가 RPM 100을 그 한 번에 넘겼다(해당 낭비 자체는
ground.py에서 노드 임베딩 캐싱으로 고침). 완전 해소 후에도 resolve_hop의 후보별 desc
임베딩처럼 반복 호출이 남아있어, 배치 루프 사이에 요청수 기준 슬라이딩 윈도우를 둔다.
"""
from __future__ import annotations
import time
import threading
import collections
from openai import OpenAI

_RPM_BUDGET = 22    # 2026-07-27: 여러 run_eval.py 프로세스를 동시에 띄울 걸 감안해 실측 100 RPM을
# 프로세스 수만큼 나눔(각자 독립된 예산 추적이라 안 나누면 합쳐서 실제 한도를 넘긴다)
_RPM_WINDOW = 60


class OpenAIEmbedder:
    def __init__(self, api_key: str, base_url: str, model: str, dim: int,
                 batch: int = 32, max_chars: int = 5000):
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.model = model
        self.dim = dim
        self.batch = batch
        # solar 임베딩 입력 한도 4000토큰. 한국어 ~0.55토큰/자라 5000자면 ~2750토큰(안전 마진).
        # 초과 시 API가 400 에러(절단 아님) → 우리가 절단. 문단 0.1%만 해당(ADR-11).
        self.max_chars = max_chars
        self._solar = "solar-embedding" in model   # -passage/-query 분기 대상
        self._req_lock = threading.Lock()
        self._req_log = collections.deque()  # [timestamp, ...] 슬라이딩 윈도우(요청 수 기준)

    def _model_for(self, is_query: bool) -> str:
        if self._solar:
            return f"{self.model}-{'query' if is_query else 'passage'}"
        return self.model

    def _throttle_request(self):
        while True:
            with self._req_lock:
                now = time.time()
                while self._req_log and self._req_log[0] < now - _RPM_WINDOW:
                    self._req_log.popleft()
                if len(self._req_log) < _RPM_BUDGET:
                    self._req_log.append(now)
                    return
                oldest = self._req_log[0]
            time.sleep(max(0.3, _RPM_WINDOW - (time.time() - oldest) + 0.2))

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        model = self._model_for(is_query)
        # 빈/공백 문자열은 Upstage가 400으로 거부한다(실측) — 자리표시자로 치환해 인덱스 보존.
        texts = [(t[:self.max_chars] if t and t.strip() else "(빈 텍스트)") for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            self._throttle_request()
            resp = self.client.embeddings.create(model=model, input=texts[i:i + self.batch])
            out.extend(d.embedding for d in resp.data)
        return out
