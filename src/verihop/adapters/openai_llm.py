"""LLMPort 구현 (OpenAI 호환 chat). Upstage solar-pro / OpenAI 공용.
implements: verihop.ports.LLMPort
schema 지정 시 JSON을 파싱해 dict 반환 (response_format json_object 시도 + 방어적 파싱).
재시도는 SDK max_retries. 비용집계는 추후(W2.2).
서드파티 import는 이 계층에서만 허용. usecase는 이 파일의 존재를 모른다.

레이트리밋 스로틀(2026-07-24): 429가 왜 나는지 추측 대신 실측했다 — Upstage가 모든 응답에
x-upstage-ratelimit-remaining-tokens·-reset-tokens 헤더로 실시간 잔여 TPM 예산을 알려준다
(실측: 100 RPM, 50,000 TPM). decompose 프롬프트 하나가 이미 ~1,650토큰(스키마+예시 포함)이라
동시성을 아무리 낮춰도 이 헤더를 안 보면 매번 추측샷이 된다. 매 응답마다 헤더를 읽어 잔여
토큰이 여유 문턱 밑이면 리셋 시각까지 자체적으로 대기 — 클라이언트 쪽 추정치 대신 서버가
알려준 실제 잔여량을 그대로 쓰므로 워커 수와 무관하게 항상 예산 안에서만 호출한다.

문제: 첫 배포 버전은 성공 응답(raw.headers)에서만 예산을 갱신했다. 그런데 429는 SDK가
raw를 반환하기 전에 APIStatusError로 던져버려서, 정작 예산 정보가 가장 필요한 실패 순간에는
헤더를 아예 못 읽고 있었다(재시도해봐도 여전히 429만 반복 — 실측: 30건 재시도 전부 429).
APIStatusError.response.headers에서도 동일하게 읽어 429 자체를 자체 재시도(최대 3회, 매
루프의 _throttle이 방금 갱신된 리셋 시각까지 대기)하도록 수정.
"""
from __future__ import annotations
import json
import re
import time
import threading
from openai import OpenAI, APIStatusError

_TOKEN_HEADROOM = 3000  # 다음 콜(추출 프롬프트는 문단 포함이라 더 큼) 하나 여유는 남기고 대기


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
        self._budget_lock = threading.Lock()
        self._remaining_tokens = None
        self._reset_at = None

    def _throttle(self):
        """직전 응답 헤더가 알려준 실시간 잔여 TPM을 보고, 부족하면 리셋까지 대기(멀티스레드 공유)."""
        with self._budget_lock:
            remaining, reset_at = self._remaining_tokens, self._reset_at
        if remaining is not None and remaining < _TOKEN_HEADROOM and reset_at:
            wait = reset_at - time.time()
            if wait > 0:
                time.sleep(wait + 0.5)

    def _update_budget(self, headers):
        try:
            remaining = int(headers.get("x-upstage-ratelimit-remaining-tokens"))
            reset_at = int(headers.get("x-upstage-ratelimit-reset-tokens"))
        except (TypeError, ValueError):
            return
        with self._budget_lock:
            self._remaining_tokens, self._reset_at = remaining, reset_at

    def complete(self, prompt: str, *, schema: dict | None = None) -> object:
        msgs = [{"role": "user", "content": prompt}]
        kwargs = {"response_format": {"type": "json_object"}} if schema is not None else {}
        raw = self._create(msgs, kwargs)
        resp = raw.parse()
        text = resp.choices[0].message.content
        return _parse_json(text) if schema is not None else text

    def _create(self, msgs, kwargs):
        for _ in range(3):                      # 429 자체 재시도(리셋 대기) + response_format 폴백 여유
            self._throttle()
            try:
                raw = self.client.chat.completions.with_raw_response.create(
                    model=self.model, temperature=self.temperature, messages=msgs, **kwargs)
            except APIStatusError as e:
                self._update_budget(e.response.headers)
                if e.status_code == 429:
                    continue                    # 다음 루프의 _throttle()이 방금 갱신된 리셋시각까지 대기
                if kwargs:                       # response_format 미지원 등 다른 오류 → 스키마 없이 폴백
                    kwargs = {}
                    continue
                raise
            self._update_budget(raw.headers)
            return raw
        raise RuntimeError("OpenAILLM._create: 429 재시도 예산(3회) 소진")
