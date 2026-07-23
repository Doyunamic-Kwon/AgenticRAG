"""EmbedderPort 구현 (OpenAI 호환 API). Upstage solar / OpenAI 공용.
implements: verihop.ports.EmbedderPort
solar-embedding은 문서=-passage, 질의=-query 변형을 쓰므로 is_query로 분기한다.
서드파티 import는 이 계층에서만 허용. usecase는 이 파일의 존재를 모른다.
test: usecase 테스트에서 tests/fakes.py로 대체됨
"""
from __future__ import annotations
from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, api_key: str, base_url: str, model: str, dim: int,
                 batch: int = 32, max_chars: int = 5000):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dim = dim
        self.batch = batch
        # solar 임베딩 입력 한도 4000토큰. 한국어 ~0.55토큰/자라 5000자면 ~2750토큰(안전 마진).
        # 초과 시 API가 400 에러(절단 아님) → 우리가 절단. 문단 0.1%만 해당(ADR-11).
        self.max_chars = max_chars
        self._solar = "solar-embedding" in model   # -passage/-query 분기 대상

    def _model_for(self, is_query: bool) -> str:
        if self._solar:
            return f"{self.model}-{'query' if is_query else 'passage'}"
        return self.model

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        model = self._model_for(is_query)
        # 빈/공백 문자열은 Upstage가 400으로 거부한다(실측) — 자리표시자로 치환해 인덱스 보존.
        texts = [(t[:self.max_chars] if t and t.strip() else "(빈 텍스트)") for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            resp = self.client.embeddings.create(model=model, input=texts[i:i + self.batch])
            out.extend(d.embedding for d in resp.data)
        return out
