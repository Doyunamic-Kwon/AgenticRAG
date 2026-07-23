#!/usr/bin/env python3
"""C 보조: 2WikiMultihopQA에서 우리 스키마·한국어 위키에 정합되는 2-hop만 뽑아 한국어 gold로.

- 2Wiki compositional(bridge) + evidences 트리플 사용 → 관계·정답 gold가 데이터에 내장.
- 관계가 우리 10종에 매핑되고, anchor·bridge 엔티티가 한국어 위키에 존재하는 것만 채택.
- 한국어 위키 intro 문단을 코퍼스 증분으로 저장(정합). 질문은 solar-pro로 한국어화.
- shortcut 필터는 생략(2wiki gold 문단이 메인 인덱스에 없음 → 통합 재인덱싱 후 적용).

usage: python3 scripts/build_2wiki_ko.py --n 10
out: data/eval/multihop_2wiki.jsonl + data/corpus_2wiki.jsonl
"""
import sys, os, json, time, argparse, urllib.request, urllib.parse, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml
from dotenv import load_dotenv

UA = {"User-Agent": "VeriHop-research/0.1 (bootcamp; keonorg@gachon.ac.kr)"}
DS = "scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35"
ROWS_API = f"https://datasets-server.huggingface.co/rows?dataset={DS}&config=default&split=validation"

# 2Wiki 관계 → 우리 스키마 (매핑 안 되는 건 폐기)
REL_MAP = {
    "place of birth": "bornIn", "place of death": "diedIn",
    "country of citizenship": "nationality", "country": "locatedIn",
    "country of origin": "locatedIn", "educated at": "studiedAt",
    "employer": "memberOf", "director": "createdBy", "performer": "createdBy",
    "composer": "createdBy", "occupation": "occupation",
}

PROMPT = """영어 2-hop 질문을 자연스러운 한국어로 번역하라. 고유명사는 아래 한국어 표기를 그대로 써라.

영어 질문: {q}
한국어 표기: {names}

다리 엔티티나 최종 답을 질문에 노출하지 말고(원문도 노출 안 함), 의미를 보존하라.
JSON만: {{"question": "한국어 질문", "answer_ko": "최종 답의 자연스러운 한국어 표기"}}"""

_cache = {}


def _fetch(url, tries=5, throttle=0.1):
    """429/일시오류 백오프 재시도. 성공 시 dict, 실패 시 None."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                d = json.load(r)
            time.sleep(throttle)
            return d
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                time.sleep(2 ** i)          # 1,2,4,8s
                continue
            return None
        except Exception:
            if i < tries - 1:
                time.sleep(1)
                continue
            return None


def wiki_get(host, params):
    params = {**params, "format": "json"}
    url = f"https://{host}/w/api.php?" + urllib.parse.urlencode(params)
    if url in _cache:
        return _cache[url]
    _cache[url] = _fetch(url)
    return _cache[url]


def ko_title(en_title):
    d = wiki_get("en.wikipedia.org", {"action": "query", "titles": en_title,
                                      "prop": "langlinks", "lllang": "ko", "redirects": 1})
    if not d:
        return None
    pg = list(d["query"]["pages"].values())[0]
    ll = pg.get("langlinks")
    return ll[0]["*"] if ll else None


def ko_intro(ko_t):
    d = wiki_get("ko.wikipedia.org", {"action": "query", "titles": ko_t, "prop": "extracts",
                                      "exintro": 1, "explaintext": 1, "redirects": 1})
    if not d:
        return None
    pg = list(d["query"]["pages"].values())[0]
    txt = (pg.get("extract") or "").strip()
    return txt[:1500] if txt else None


def fetch_rows(pages, split="validation"):
    """문제: 이전 실행에서 한 페이지 실패 시 전체 스캔을 중단해 12,576행 중 3,600행에서 멈췄었다
    (validation split 전체 크기를 몰랐던 게 원인 — HF datasets-server /size로 사전 확인 가능했음).
    수정: 실패한 페이지는 건너뛰고 계속 진행(연속 실패 5회 초과 시에만 중단, API 전체 장애 대비)."""
    api = f"https://datasets-server.huggingface.co/rows?dataset={DS}&config=default&split={split}"
    rows, consecutive_fail = [], 0
    for off in range(0, pages * 100, 100):
        d = _fetch(f"{api}&offset={off}&length=100", throttle=0.5)
        if not d:
            consecutive_fail += 1
            print(f"  rows fetch 실패(off={off}, 연속 {consecutive_fail}) — 건너뛰고 계속")
            if consecutive_fail >= 5:
                print(f"  연속 5회 실패 → 중단(off={off})"); break
            continue
        consecutive_fail = 0
        got = d.get("rows", [])
        rows += [r["row"] for r in got]
        if len(got) < 100:
            break
    return rows


def make_llm():
    load_dotenv(ROOT / ".env")
    c = yaml.safe_load(open(ROOT / "configs/settings.yaml", encoding="utf-8"))
    lp = c["providers"]["llm"]; lc = c["llm"][lp]
    key = {"upstage": "UPSTAGE_API_KEY", "openai": "OPENAI_API_KEY"}[lp]
    from verihop.adapters.openai_llm import OpenAILLM
    return OpenAILLM(os.environ[key], lc["base_url"], lc["model"],
                     c["llm"].get("temperature", 0.0), c["llm"].get("max_retries", 2)), c


def load_relations():
    r = yaml.safe_load(open(ROOT / "configs/relations.yaml", encoding="utf-8"))
    return {x["name"]: (x["head"], x["tail"]) for x in r["relations"]}


def main(n, pages):
    llm, _ = make_llm()
    rel = load_relations()
    rows = fetch_rows(pages)
    comp = [r for r in rows if r.get("type") == "compositional"]
    print(f"2Wiki 로드 {len(rows)}행 → compositional {len(comp)}. 목표 {n}문항.")

    gold, corpus_add, seen_pid, seen_fact = [], [], set(), set()
    checked = mapped = ko_ok = 0
    for r in comp:
        if len(gold) >= n:
            break
        ev = r["evidences"]
        if len(ev["relation"]) != 2:
            continue
        checked += 1
        r0, r1 = ev["relation"]
        if r0 not in REL_MAP or r1 not in REL_MAP:
            continue
        mapped += 1
        m0, m1 = REL_MAP[r0], REL_MAP[r1]
        anchor_en, bridge_en = ev["fact"][0], ev["entity"][0]
        answer_en = ev["entity"][1]
        if answer_en in (bridge_en, anchor_en):       # 답==다리/앵커 → 깨진 사슬 (예: diedIn 답이 인물)
            continue
        fkey = (bridge_en, m1, answer_en)             # hop2 사실 중복 제거 (같은 인물 출생지 반복 방지)
        if fkey in seen_fact:
            continue
        anchor_ko, bridge_ko = ko_title(anchor_en), ko_title(bridge_en)
        if not anchor_ko or not bridge_ko:            # 두 evidence 문서의 한국어 페이지 필요
            continue
        a_intro, b_intro = ko_intro(anchor_ko), ko_intro(bridge_ko)
        if not a_intro or not b_intro:
            continue
        ko_ok += 1
        answer_ko = ko_title(answer_en)               # 있으면 사용, 없으면 LLM 표기

        names = ", ".join(f"{en}={ko}" for en, ko in
                          [(anchor_en, anchor_ko), (bridge_en, bridge_ko)] if ko)
        try:
            out = llm.complete(PROMPT.format(q=r["question"], names=names), schema=True)
        except Exception as e:
            print(f"  LLM skip: {e}"); continue
        if not isinstance(out, dict) or not out.get("question"):
            continue
        ans_ko = answer_ko or out.get("answer_ko") or answer_en

        pid_a, pid_b = f"2wiki::{anchor_ko}", f"2wiki::{bridge_ko}"
        for pid, title, txt in [(pid_a, anchor_ko, a_intro), (pid_b, bridge_ko, b_intro)]:
            if pid not in seen_pid:
                seen_pid.add(pid)
                corpus_add.append({"paragraph_id": pid, "doc_title": title, "text": txt})

        seen_fact.add(fkey)
        gold.append({
            "qid": f"mh2w_{len(gold):04d}",
            "question": out["question"].strip(),
            "answer": ans_ko,
            "answer_aliases": [ans_ko, answer_en],
            "structure": "CHAIN",
            "source": "2wiki",
            "hops": [
                {"hop": 1, "start": anchor_ko, "start_type": rel[m0][0], "relation": m0,
                 "direction": "fwd", "target": bridge_ko, "target_type": rel[m0][1],
                 "target_aliases": [bridge_ko, bridge_en],
                 "evidence_paragraph": pid_a, "source_qa": r["_id"]},
                {"hop": 2, "start": "{1}", "start_type": rel[m1][0], "relation": m1,
                 "direction": "fwd", "target": ans_ko, "target_type": rel[m1][1],
                 "target_aliases": [ans_ko, answer_en],
                 "evidence_paragraph": pid_b, "source_qa": r["_id"]},
            ],
        })
        print(f"  수집 {len(gold)}/{n}: {out['question'].strip()[:50]}... ({r0}/{r1})", flush=True)

    (ROOT / "data/eval").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "data/eval/multihop_2wiki.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(ROOT / "data/corpus_2wiki.jsonl", "w", encoding="utf-8") as f:
        for cadd in corpus_add:
            f.write(json.dumps(cadd, ensure_ascii=False) + "\n")
    print(f"\n완료: gold {len(gold)}문항, 코퍼스 증분 {len(corpus_add)}문단")
    print(f"  통과 흐름: 관계매핑 {mapped}/{checked} → 한국어위키 정합 {ko_ok} → 채택 {len(gold)}")
    print("  ※ 통합 재인덱싱(코퍼스 + corpus_2wiki) 후 shortcut 필터·eval 적용.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pages", type=int, default=15, help="2Wiki rows 페이지 수 (×100)")
    a = ap.parse_args()
    main(a.n, a.pages)
