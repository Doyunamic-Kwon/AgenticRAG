"""LLMPort 구현 (OpenAI 호환 chat). Upstage solar-pro / OpenAI 공용.
implements: verihop.ports.LLMPort
schema 지정 시 JSON을 파싱해 dict 반환 (response_format json_object 시도 + 방어적 파싱).
재시도는 SDK max_retries. 비용집계는 추후(W2.2).
서드파티 import는 이 계층에서만 허용. usecase는 이 파일의 존재를 모른다.
"""
from __future__ import annotations
import json
import re
from openai import OpenAI


def _parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)   # 코드펜스/설명 섞여도 첫 객체 추출
        if m:
            return json.loads(m.group(0))
        raise


class OpenAILLM:
    def __init__(self, api_key: str, base_url: str, model: str,
                 temperature: float = 0.0, max_retries: int = 2, timeout: float = 60):
        # timeout: 요청당 상한(초). 없으면 solar가 매달려 배치가 무한 정지(실측).
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             max_retries=max_retries, timeout=timeout)
        self.model = model
        self.temperature = temperature

    def complete(self, prompt: str, *, schema: dict | None = None) -> object:
        msgs = [{"role": "user", "content": prompt}]
        kwargs = {"response_format": {"type": "json_object"}} if schema is not None else {}
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, messages=msgs, **kwargs)
        except Exception:
            if kwargs:                          # response_format 미지원 모델 폴백
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, messages=msgs)
            else:
                raise
        text = resp.choices[0].message.content
        return _parse_json(text) if schema is not None else text
