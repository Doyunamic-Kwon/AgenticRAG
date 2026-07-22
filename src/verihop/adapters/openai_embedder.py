"""EmbedderPort 구현 (OpenAI 호환 API). Upstage solar / OpenAI 공용.
implements: verihop.ports.EmbedderPort
solar-embedding은 문서=-passage, 질의=-query 변형을 쓰므로 is_query로 분기한다.
서드파티 import는 이 계층에서만 허용. usecase는 이 파일의 존재를 모른다.
test: usecase 테스트에서 tests/fakes.py로 대체됨
"""
from __future__ import annotations
from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str, base_url: str, model: str, dim: int, batch: int = 100):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dim = dim
        self.batch = batch
        self._solar = "solar-embedding" in model   # -passage/-query 분기 대상

    def _model_for(self, is_query: bool) -> str:
        if self._solar:
            return f"{self.model}-{'query' if is_query else 'passage'}"
        return self.model

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        model = self._model_for(is_query)
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = self.client.embeddings.create(model=model, input=texts[i:i + self.batch])
            out.extend(d.embedding for d in resp.data)
        return out
