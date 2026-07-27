"""LLMPort 구현 (OpenAI 호환 chat). Upstage solar-pro / OpenAI 공용.
implements: verihop.ports.LLMPort
schema 지정 시 JSON을 파싱해 dict 반환 (response_format json_object 시도 + 방어적 파싱).
비용집계는 추후(W2.2).
서드파티 import는 이 계층에서만 허용. usecase는 이 파일의 존재를 모른다.

레이트리밋 스로틀(2026-07-24): 429 원인을 단계적으로 실측했다.
1차 시도 — 서버가 매 응답에 실어주는 x-upstage-ratelimit-remaining-tokens/-reset-tokens
  헤더(문서상 100 RPM/50,000 TPM)를 그대로 신뢰해 스로틀. 그런데 실제 파이프라인 콜을
  10회 넘게 연속으로 찍어봐도 remaining이 49998에서 전혀 안 줄었다 — 이 헤더 자체가
  실사용량을 반영 안 함을 확인(버그인지 별도 계정 단위 카운터인지는 불명, 신뢰 불가로 판단).
2차 확인 — 단발성 대형 프롬프트·JSON 모드·4스레드 동시실행을 개별/조합으로 다 재현
  시도했지만 전부 성공. 반면 실제 4-worker 파이프라인 재현(문항당 여러 콜을 스레드가
  200초 넘게 이어가는 패턴)은 4개 중 3개가 429로 재현됨 — 단발 호출 특성이 아니라
  "여러 스레드가 오랜 시간에 걸쳐 콜을 계속 이어갈 때의 누적 총량"이 문제라는 뜻이다.
결론: 서버 헤더를 못 믿으므로 클라이언트가 직접 사용량을 실측(resp.usage.total_tokens)해
슬라이딩 윈도우로 자체 계산한다. 호출 전 프롬프트 길이로 예상 토큰을 추정해 예산에서
미리 예약(다른 스레드가 그 사이 예산을 초과 소비하지 않도록)하고, 응답이 오면 실제
사용량으로 보정한다. 문서상 한도(50,000 TPM)보다 상당히 보수적인 자체 예산을 쓴다 —
문서 자체도 이번에 신뢰가 흔들렸고, 우리 쪽 추정치도 어차피 근사값이라 여유가 필요하다.
"""
from __future__ import annotations
import json
import re
import time
import threading
import collections
from openai import OpenAI, APIStatusError

_TPM_BUDGET = 30000    # Upstage 기준값(50,000의 60%). gpt-4o-mini(2026-07-27) 전환 후 실측
# 헤더는 5,000 RPM/4,000,000 TPM으로 훨씬 넉넉하고 값도 정상 갱신됨(Upstage와 달리 신뢰 가능)
# — provider별로 예산이 달라 아래에서 model 문자열로 분기해 override.
_TPM_BUDGET_BY_MODEL = {"gpt-4o-mini": 1000000}   # 실측치(4M)의 25% — 여러 run_eval.py 프로세스를
# 동시에 띄울 걸 감안(각자 독립된 예산 추적이라 프로세스 수만큼 나눠 잡아야 실제 한도 안에 든다)
_TPM_WINDOW = 60        # 초. Upstage 문서상 interval=minute
_CHARS_PER_TOKEN = 2.2  # 실측(decompose 3738자→1453 prompt-tokens, 추출 6132자→2622 prompt-tokens)
_COMPLETION_BUFFER = 400  # 응답 토큰 여유분(추정치에 더함)


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
        # SDK 자체 재시도(지수백오프, 우리 스로틀과 무관하게 동작)는 끄고, max_retries는
        # 아래 _create()의 자체 재시도 횟수로 쓴다 — 이중 재시도가 겹치면 예측 불가능해진다.
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self._max_retries = max(max_retries, 3)
        self._tpm_budget = _TPM_BUDGET_BY_MODEL.get(model, _TPM_BUDGET)
        self._budget_lock = threading.Lock()
        self._usage_log = collections.deque()  # [[timestamp, tokens], ...] 슬라이딩 윈도우(실측치)

    def _wait_until_room(self):
        """로그(내 예약 슬롯 포함)를 정리하고, 예산을 넘겼으면 가장 오래된 항목이 윈도우를
        벗어날 때까지 대기. 새 항목을 추가하지 않는 조회/대기 전용 — _create가 슬롯을 관리한다."""
        while True:
            with self._budget_lock:
                now = time.time()
                while self._usage_log and self._usage_log[0][0] < now - _TPM_WINDOW:
                    self._usage_log.popleft()
                used = sum(t for _, t in self._usage_log)
                if used <= self._tpm_budget or not self._usage_log:
                    return
                oldest_ts = self._usage_log[0][0]
            time.sleep(max(0.3, _TPM_WINDOW - (time.time() - oldest_ts) + 0.2))

    def complete(self, prompt: str, *, schema: dict | None = None) -> object:
        msgs = [{"role": "user", "content": prompt}]
        kwargs = {"response_format": {"type": "json_object"}} if schema is not None else {}
        resp = self._create(msgs, kwargs)
        text = resp.choices[0].message.content
        return _parse_json(text) if schema is not None else text

    def _create(self, msgs, kwargs):
        # 문제(2026-07-24): 이전 버전은 재시도마다 _reserve()를 새로 불러 로그에 항목을 계속
        # 추가했다 — 429가 나서 재시도할수록 "사용 중"으로 잡히는 가짜 예약이 계속 쌓여
        # 스스로 예산을 고갈시키는 악순환에 빠짐(실측: 1개 호출이 1시간 넘게 대기하다 연결
        # 오류로 끝남). 이 논리 호출 하나당 슬롯을 정확히 하나만 예약해 재시도 내내 재사용한다.
        prompt_chars = sum(len(m["content"]) for m in msgs)
        estimated = int(prompt_chars / _CHARS_PER_TOKEN) + _COMPLETION_BUFFER
        entry = [time.time(), estimated]
        with self._budget_lock:
            self._usage_log.append(entry)          # 60초 뒤 창밖으로 자연 소멸, 별도 해제 불필요
        for _ in range(self._max_retries):
            self._wait_until_room()
            try:
                raw = self.client.chat.completions.with_raw_response.create(
                    model=self.model, temperature=self.temperature, messages=msgs, **kwargs)
            except APIStatusError as e:
                if e.status_code == 429:
                    entry[1] = estimated * 2       # 과소추정 방지 — 다음 대기 판단을 보수적으로
                    continue
                entry[1] = 0                        # 실제로 안 나간 예약분 취소
                if kwargs:                           # response_format 미지원 등 다른 오류 → 폴백
                    kwargs = {}
                    continue
                raise
            resp = raw.parse()
            entry[1] = resp.usage.total_tokens if resp.usage else estimated
            return resp
        raise RuntimeError(f"OpenAILLM._create: 429 재시도 예산({self._max_retries}회) 소진")
