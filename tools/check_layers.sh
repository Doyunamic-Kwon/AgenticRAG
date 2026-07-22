#!/usr/bin/env bash
# 계층 의존 규칙 자동 검증 (06 / ADR-10). 커밋 전 통과가 done 기준.
# 의존 방향: domain <- usecases <- adapters <- bootstrap/apps/scripts
# ponytail: grep 휴리스틱(진짜 import 그래프 아님). 위반 오탐 시 규칙 조정.
set -u
root="src/verihop"
fail=0

check() {  # $1 설명  $2 위반이면 비어있지 않을 grep
  out=$(eval "$2" || true)
  if [ -n "$out" ]; then printf '❌ %s\n%s\n' "$1" "$out"; fail=1; else printf '✅ %s\n' "$1"; fi
}

check "usecases가 adapters를 import 안 함" \
  "grep -rn 'verihop.adapters' $root/usecases/ 2>/dev/null"
check "usecases가 서드파티를 직접 import 안 함 (ports 경유)" \
  "grep -rnE 'import (openai|chromadb|networkx)|from (openai|chromadb|networkx)' $root/usecases/ 2>/dev/null"
check "domain이 verihop.models 외 verihop을 import 안 함" \
  "grep -rn 'from verihop\.' $root/domain/ 2>/dev/null | grep -v 'from verihop.models'"
check "domain이 서드파티를 import 안 함" \
  "grep -rnE 'import (openai|chromadb|networkx|numpy|pydantic)|from (openai|chromadb|networkx|numpy|pydantic)' $root/domain/ 2>/dev/null"

exit $fail
