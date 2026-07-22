#!/usr/bin/env python3
"""W1.6 멀티홉 gold 평가셋 구축 (05 gold 스키마).

브릿지 쌍(W0.1) → solar-pro가 사슬 타당성·관계를 심판 → shortcut 필터 → data/eval/multihop.jsonl
- W0.1 relation 태그는 분포용(문자열 매칭)이라 문항별 gold로 못 쓴다. LLM이 스키마에서 관계를 직접
  분류하고, 억지 사슬(우연한 단어 일치)은 reject한다. type은 관계에서 파생(스키마 정합).
- shortcut 필터: Baseline(원질의 1회 검색)이 최종 정답 문단을 top-10에 잡으면 탈락(너무 쉬움).
금지: 자체 그래프 경로 역생성(순환 평가). 여기선 KorQuAD 브릿지만 사용.

usage: python3 scripts/build_multihop_set.py --n 100
out: data/eval/multihop.jsonl  (자연성 최종 수검수는 사람이 후속)
"""
import sys, os, json, re, random, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import faiss
import yaml
from dotenv import load_dotenv

KEY_ENV = {"upstage": "UPSTAGE_API_KEY", "openai": "OPENAI_API_KEY", "naver": "NAVER_CLOVA_API_KEY"}

PROMPT = """아래 두 단일홉 QA가 자연스러운 2-hop 추론 사슬을 이루는지 엄격히 판단하라.

관계 스키마(반드시 이 이름 중에서만 선택):
bornIn(태어난 곳), diedIn(죽은 곳), locatedIn(상위 지역), capitalOf(수도), nationality(국적),
studiedAt(출신 학교), memberOf(소속 단체), createdBy(만든 사람), teacherOf(스승/제자), occupation(직업)

[hop1] 질문: {qa}
       답: {ba}
[hop2] 질문: {qb}
       답: {bb}
(hop2 질문은 hop1의 답 '{bridge}'를 핵심 대상으로 포함한다)

규칙:
- hop1의 답 '{bridge}'가 hop2 질문의 주어로 자연스럽게 이어져야 한다.
- hop1과 hop2가 각각 위 스키마의 관계 하나로 명확히 분류돼야 한다. 억지로 끼워맞추지 마라.
- **답 타입 정합**: 각 관계의 답이 그 관계의 대상 타입과 맞아야 한다.
  bornIn/diedIn/locatedIn/capitalOf/nationality의 답은 지명, createdBy/teacherOf는 사람,
  studiedAt/memberOf는 단체, occupation은 직업이어야 한다. 날짜·숫자·수식어구가 답이면 valid=false.
  (예: '{bridge}'가 안철수인데 hop1을 memberOf로 분류하면 틀림 — 사람은 memberOf의 답이 될 수 없다.)
- 우연한 단어 일치이거나, 사슬이 어색하거나, 최종 답이 유일하지 않으면 valid=false.
- valid=true면: 다리 엔티티('{bridge}')를 직접 언급하지 않는 자연스러운 합친 질문,
  hop1의 앵커(질문의 주어 고유명사), 두 hop의 관계명.

JSON만 출력:
{{"valid": true/false, "question": "합친 질문", "anchor": "hop1 앵커",
  "relation1": "스키마 관계명", "relation2": "스키마 관계명"}}"""


def make_clients(llm_provider=None):
    load_dotenv(ROOT / ".env")
    c = yaml.safe_load(open(ROOT / "configs/settings.yaml", encoding="utf-8"))
    ep = c["providers"]["embedding"]; ec = c["embedding"][ep]
    lp = llm_provider or c["providers"]["llm"]; lc = c["llm"][lp]
    from verihop.adapters.openai_embedder import OpenAIEmbedder
    from verihop.adapters.openai_llm import OpenAILLM
    embedder = OpenAIEmbedder(os.environ[KEY_ENV[ep]], ec["base_url"], ec["model"], ec["dim"],
                              batch=c["embedding"].get("batch", 32),
                              max_chars=c["embedding"].get("max_chars", 5000))
    llm = OpenAILLM(os.environ[KEY_ENV[lp]], lc["base_url"], lc["model"],
                    c["llm"].get("temperature", 0.0), c["llm"].get("max_retries", 2))
    return c, embedder, llm


def load_relations():
    r = yaml.safe_load(open(ROOT / "configs/relations.yaml", encoding="utf-8"))
    return {x["name"]: (x["head"], x["tail"]) for x in r["relations"]}


def aliases(ans):
    a = ans.strip()
    m = re.match(r"^(.+?)\s*[(（](.+?)[)）]\s*$", a)
    return [m.group(1).strip(), m.group(2).strip()] if m else [a]


def load_qid2pid():
    text2pid = {json.loads(l)["text"]: json.loads(l)["paragraph_id"]
                for l in open(ROOT / "data/corpus.jsonl", encoding="utf-8")}
    qid2pid = {}
    for split in ["train", "dev"]:
        data = json.load(open(ROOT / f"data/raw/KorQuAD_v1.0_{split}.json", encoding="utf-8"))["data"]
        for doc in data:
            for para in doc["paragraphs"]:
                pid = text2pid.get(para["context"].strip())
                if pid:
                    for qa in para["qas"]:
                        qid2pid[qa["id"]] = pid
    return qid2pid


def hop(idx, start, start_type, relation, target, target_type, pid, src):
    return {"hop": idx, "start": start, "start_type": start_type, "relation": relation,
            "direction": "fwd", "target": target, "target_type": target_type,
            "target_aliases": aliases(target), "evidence_paragraph": pid, "source_qa": src}


def main(n, max_attempts, llm_provider):
    c, embedder, llm = make_clients(llm_provider)
    rel = load_relations()
    qid2pid = load_qid2pid()
    index = faiss.read_index(str(ROOT / c["vector"]["dir"] / "faiss.index"))
    ids = [json.loads(l) for l in open(ROOT / c["vector"]["dir"] / "ids.jsonl", encoding="utf-8")]

    bridges = [json.loads(l) for l in open(ROOT / "data/w0/bridge_candidates.jsonl", encoding="utf-8")]
    # W0.1 태그로 1차 선별(둘 다 스키마)만 하고, 문항별 관계는 LLM이 다시 판정한다.
    pool = [b for b in bridges if b["relation_a"] in rel and b["relation_b"] in rel]
    seen, uniq = set(), []
    for b in pool:
        k = b["qa_b"]["qid"]
        if k not in seen:
            seen.add(k); uniq.append(b)
    random.seed(0); random.shuffle(uniq)
    print(f"1차 풀 {len(pool)} → 고유 최종질문 {len(uniq)}. 목표 {n}문항.")

    gold, attempts, d_short, d_invalid, d_schema = [], 0, 0, 0, 0
    for b in uniq:
        if len(gold) >= n or attempts >= max_attempts:
            break
        attempts += 1
        qa, qb = b["qa_a"], b["qa_b"]
        pid_a, pid_b = qid2pid.get(qa["qid"]), qid2pid.get(qb["qid"])
        if not pid_a or not pid_b:
            continue
        try:
            out = llm.complete(PROMPT.format(qa=qa["question"], ba=qa["answer"],
                                             bridge=b["bridge_entity"], qb=qb["question"], bb=qb["answer"]),
                               schema=True)
        except Exception as e:
            print(f"  LLM skip: {e}"); continue
        if not isinstance(out, dict) or not out.get("valid") or not out.get("question"):
            d_invalid += 1; continue
        ra, rb = out.get("relation1"), out.get("relation2")
        if ra not in rel or rb not in rel:               # LLM이 스키마 밖 관계 → 폐기
            d_schema += 1; continue
        question = out["question"].strip()

        qv = np.asarray(embedder.embed([question], is_query=True), dtype="float32")
        faiss.normalize_L2(qv)
        _, I = index.search(qv, 10)
        if pid_b in {ids[j]["paragraph_id"] for j in I[0]}:   # Baseline이 잡음 → 너무 쉬움
            d_short += 1; continue

        gold.append({
            "qid": f"mh_{len(gold):04d}",
            "question": question,
            "answer": qb["answer"],
            "answer_aliases": aliases(qb["answer"]),
            "structure": "CHAIN",
            "hops": [
                hop(1, out.get("anchor", "").strip(), rel[ra][0], ra,
                    b["bridge_entity"], rel[ra][1], pid_a, qa["qid"]),
                hop(2, "{1}", rel[rb][0], rb, qb["answer"], rel[rb][1], pid_b, qb["qid"]),
            ],
        })
        if len(gold) % 10 == 0:
            print(f"  수집 {len(gold)}/{n} (시도 {attempts}, short {d_short}, invalid {d_invalid}, off-schema {d_schema})", flush=True)

    out_path = ROOT / "data/eval/multihop.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"\n완료: {len(gold)}문항 → {out_path}")
    print(f"  시도 {attempts} | shortcut {d_short} | invalid {d_invalid} | off-schema {d_schema}")
    print("  ※ direction=fwd 가정(초판). 자연성 최종 수검수는 사람이 후속(05).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-attempts", type=int, default=700)
    ap.add_argument("--llm-provider", default=None, help="settings 기본 대신 심판 LLM 지정 (openai 등)")
    a = ap.parse_args()
    main(a.n, a.max_attempts, a.llm_provider)
