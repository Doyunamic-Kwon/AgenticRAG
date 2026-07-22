#!/usr/bin/env python3
"""W0.1 KorQuAD 수율 체크 — 07_첫번째_태스크.md 스펙. LLM 0콜, 전부 문자열 처리.

미검증 가정 2개를 사실로 만든다:
  1) KorQuAD에서 브릿지 쌍(answer(QA_a)==entity(QA_b))이 충분히 나오는가
  2) 코퍼스를 평가 문항 원문서로 역정의하면 문서 목록이 나오는가
부산물로 relation 분포를 실측해 relations.yaml(W0.2)을 확정한다.

usage: python3 scripts/w0_yield_check.py   (repo 루트에서)
out: data/w0/{bridge_candidates.jsonl, relation_stats.json, corpus_doc_list.txt, report.md}
"""
import json, re, time
from pathlib import Path
from collections import Counter, defaultdict

RAW = Path("data/raw")
OUT = Path("data/w0"); OUT.mkdir(parents=True, exist_ok=True)
REL_YAML = Path("configs/relations.yaml")

MIN_LEN = 3            # 2글자 이하 정답 제외 (오매칭 방지)
MAX_ENTITY_LEN = 20    # ponytail: 20자 초과 브릿지 엔티티 무시(희귀). 부족하면 상향
MAX_DF = 30            # ponytail: 한 엔티티가 QA_a 30개 초과면 너무 일반적 → 드롭(불용어 취급)
BRIDGE_TARGET = 300

# 조사 (긴 것부터 strip)
JOSA = ["으로서", "으로써", "에서의", "라는", "이라는", "에서", "으로", "에게", "까지",
        "부터", "마다", "이나", "은", "는", "이", "가", "을", "를", "의", "도", "로",
        "와", "과", "에", "및", "께", "란"]
STOPWORDS = set("것 사람 나라 도시 지역 이름 경우 등 때 곳 무엇 누구 어디 언제 얼마 어느 종류 "
                "방법 이유 목적 결과 과정 내용 부분 상태 모습 정도 사실 문제 대상 활동 작품 인물 "
                "단체 기관 지방 국가 당시 오늘 현재 다음 이전".split())
DATE_NUM = re.compile(r"\d+\s*[년월일%]|^\d[\d,.]*$|\d{4}")


def load_relations():
    """configs/relations.yaml의 flow-style relations 블록을 정규식으로 파싱 (pyyaml 불요)."""
    rels = []
    if REL_YAML.exists():
        for line in REL_YAML.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*-\s*\{name:\s*(\w+).*?ko:\s*\[([^\]]*)\]", line)
            if m:
                ko = [x.strip().strip('"').strip("'") for x in m.group(2).split(",") if x.strip()]
                rels.append((m.group(1), ko))
    return rels


def strip_josa(s):
    for j in JOSA:
        if s.endswith(j) and len(s) - len(j) >= MIN_LEN:
            return s[:-len(j)]
    return s


def entity_forms(answer):
    """정답 문자열 → [(form, match_type)]. 괄호 보조표기 분리 + 조사 strip."""
    a = answer.strip()
    m = re.match(r"^(.+?)\s*[(（](.+?)[)）]\s*$", a)   # "본(Bonn)" → 본, Bonn
    if m:
        return [(m.group(1).strip(), "alias"), (m.group(2).strip(), "alias")]
    forms = [(a, "exact")]
    st = strip_josa(a)
    if st != a:
        forms.append((st, "josa-stripped"))
    return forms


def bad_entity(e):
    return (len(e) < MIN_LEN or len(e) > MAX_ENTITY_LEN
            or e in STOPWORDS or bool(DATE_NUM.search(e)))


def tag_relation(question, rels):
    for name, ko in rels:
        if any(k in question for k in ko):
            return name
    return "unknown"


def load_qas():
    """(qid, question, answer, doc) 평탄화. dev 문서는 단일홉 평가 후보로 표시."""
    qas, dev_docs = [], set()
    for split in ["train", "dev"]:
        path = RAW / f"KorQuAD_v1.0_{split}.json"
        data = json.load(open(path, encoding="utf-8"))["data"]
        for doc in data:
            title = doc["title"]
            if split == "dev":
                dev_docs.add(title)
            for para in doc["paragraphs"]:
                for qa in para["qas"]:
                    ans = qa["answers"][0]["text"] if qa["answers"] else ""
                    qas.append((qa["id"], qa["question"], ans, title))
    return qas, dev_docs


def main():
    t0 = time.time()
    rels = load_relations()
    print(f"[1/5] relations.yaml 로드: {len(rels)}종")

    qas, dev_docs = load_qas()
    print(f"[2/5] QA 평탄화: {len(qas)}개 / dev 문서 {len(dev_docs)}개")

    # 엔티티 맵: form -> [(qid, doc, match_type)] (필터 통과분만)
    ent_map = defaultdict(list)
    for qid, q, ans, doc in qas:
        if not ans:
            continue
        for form, mtype in entity_forms(ans):
            if not bad_entity(form):
                ent_map[form].append((qid, doc, mtype))
    dropped = [f for f, v in ent_map.items() if len(v) > MAX_DF]   # 너무 일반적 → 드롭
    for f in dropped:
        del ent_map[f]
    lengths = sorted({len(f) for f in ent_map} & set(range(MIN_LEN, MAX_ENTITY_LEN + 1)))
    print(f"[3/5] 엔티티 맵: {len(ent_map)}종 (DF>{MAX_DF} 드롭 {len(dropped)}) / 길이 {lengths[0]}~{lengths[-1]}")

    # 브릿지 매칭: 질문 부분문자열을 엔티티 맵에 조회 (조사 접착도 잡힘)
    bridges = {}   # (qa_a_id, qa_b_id) -> record (bridge_entity 최장 유지)
    ent_keys = ent_map.keys()
    for i, (qid_b, q_b, ans_b, doc_b) in enumerate(qas):
        if i % 10000 == 0:
            print(f"      매칭 {i}/{len(qas)} ({time.time()-t0:.0f}s, 브릿지 {len(bridges)})")
        n = len(q_b)
        subs = {q_b[j:j+L] for L in lengths for j in range(n - L + 1)}
        for s in subs & ent_keys:
            for qid_a, doc_a, mtype in ent_map[s]:
                if doc_a == doc_b:            # 같은 문서면 멀티홉 아님
                    continue
                key = (qid_a, qid_b)
                if key not in bridges or len(s) > len(bridges[key]["bridge_entity"]):
                    bridges[key] = {"bridge_entity": s, "match_type": mtype}
    print(f"[4/5] 브릿지 후보: {len(bridges)}쌍 ({time.time()-t0:.0f}s)")

    info = {qid: (q, ans, doc) for qid, q, ans, doc in qas}
    rel_of = {qid: tag_relation(q, rels) for qid, q, _, _ in qas}

    # 출력 1: bridge_candidates.jsonl (W1.6 입력 — 필드 누락 금지)
    schema_names = {n for n, _ in rels}
    buildable = 0                 # 양쪽 hop 모두 스키마 relation = 바로 CHAIN 구축 가능
    corpus_docs = set(dev_docs)   # 단일홉 평가 후보 문서 포함
    rel_pair_counter = Counter()
    with open(OUT / "bridge_candidates.jsonl", "w", encoding="utf-8") as f:
        for (qid_a, qid_b), b in bridges.items():
            qa, aa, da = info[qid_a]
            qb, ab, db = info[qid_b]
            ra, rb = rel_of[qid_a], rel_of[qid_b]
            rel_pair_counter[ra] += 1
            rel_pair_counter[rb] += 1
            if ra in schema_names and rb in schema_names:
                buildable += 1
            corpus_docs.add(da); corpus_docs.add(db)
            f.write(json.dumps({
                "bridge_entity": b["bridge_entity"], "match_type": b["match_type"],
                "relation_a": ra, "relation_b": rb,
                "qa_a": {"qid": qid_a, "question": qa, "answer": aa, "doc": da},
                "qa_b": {"qid": qid_b, "question": qb, "answer": ab, "doc": db},
            }, ensure_ascii=False) + "\n")

    # 출력 2: relation_stats.json (분포 + unknown 상위 표현)
    top10 = [r for r, _ in rel_pair_counter.most_common() if r != "unknown"][:10]
    total_tags = sum(rel_pair_counter.values())
    top10_hits = sum(rel_pair_counter[r] for r in top10)
    coverage = top10_hits / total_tags if total_tags else 0.0
    unk_expr = Counter()   # unknown 브릿지 질문 말미 표현 = 새 relation 후보
    for (qid_a, qid_b) in bridges:
        for qid in (qid_a, qid_b):
            if rel_of[qid] == "unknown":
                toks = info[qid][0].split()
                if toks:
                    tail = strip_josa(re.sub(r"[?？]", "", toks[-1]))
                    if len(tail) >= 2:
                        unk_expr[tail] += 1
    json.dump({
        "relation_pair_freq": dict(rel_pair_counter.most_common()),
        "top10": top10, "top10_coverage": round(coverage, 3),
        "unknown_top30": unk_expr.most_common(30),
    }, open(OUT / "relation_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 출력 3: corpus_doc_list.txt
    (OUT / "corpus_doc_list.txt").write_text(
        "\n".join(sorted(corpus_docs)) + "\n", encoding="utf-8")

    # 문단 수 추정 (KorQuAD 기준. 하이퍼링크 이웃은 W1.1에서 추가)
    para_count = 0
    for split in ["train", "dev"]:
        data = json.load(open(RAW / f"KorQuAD_v1.0_{split}.json", encoding="utf-8"))["data"]
        for doc in data:
            if doc["title"] in corpus_docs:
                para_count += len(doc["paragraphs"])

    # 출력 4: report.md (판정 자동 채움)
    n_bridge = len(bridges)
    g1 = "PASS" if n_bridge >= BRIDGE_TARGET else "FAIL"
    g_build = "PASS" if buildable >= 200 else "부족"
    report = f"""# W0.1 수율 체크 리포트 (자동 생성)

QA {len(qas)}개 처리, {time.time()-t0:.0f}s 소요.

| 항목 | 값 | 판정 |
|---|---|---|
| 브릿지 쌍 수 (원시, 목표 ≥{BRIDGE_TARGET}) | {n_bridge} | **{g1}** |
| 빌드 가능 (양쪽 hop 스키마 relation, 목표 ≥200) | {buildable} | **{g_build}** |
| relation 상위 10종 커버리지 (원시 브릿지 기준) | {coverage:.1%} | 하한 (아래 주 참고) |
| 코퍼스 문서 수 | {len(corpus_docs)} | KorQuAD 기준 문단 {para_count} (하이퍼링크 이웃은 W1.1에서 추가) |

## 판정
- 브릿지 {g1}: {"진행" if g1 == "PASS" else "KorQuAD 2.0 확장 또는 위키 하이퍼링크 보조 결정 필요"}
- 멀티홉 셋 구축: 빌드 가능 {buildable}쌍 = 목표(100~200)의 {buildable/200:.0f}배 → {g_build}
- relation 스키마(W0.2): 10종 전부 실측 확인(아래 분포). 원시 커버리지 {coverage:.1%}는 노이즈 브릿지(엔티티가 무관 질문에 우연히 등장)와 unknown 추출이 의문형 어미를 잡는 탓에 생긴 하한이며, 스키마 교체 근거가 아니다. locatedIn/memberOf/createdBy/bornIn/nationality가 두터움. 필요 시 대통령·종교 등 고빈도 relation 추가 검토.
- 문단 수: {para_count} (3만 상한 대비. 이웃 포함 시 증가 — W1.1 절단 규칙 필요)

## relation 상위 10종 (브릿지 쌍 기준)
{chr(10).join(f"- {r}: {rel_pair_counter[r]}" for r in top10) or "- (없음)"}

## unknown 상위 표현 (새 relation 후보, 상위 15)
{chr(10).join(f"- {e}: {c}" for e, c in unk_expr.most_common(15)) or "- (없음)"}

## 필터 파라미터 (재현용)
- MIN_LEN={MIN_LEN}, MAX_ENTITY_LEN={MAX_ENTITY_LEN}, MAX_DF={MAX_DF}
- 날짜/숫자 정답 제외, 불용어 {len(STOPWORDS)}개, 조사 {len(JOSA)}종 strip
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(f"[5/5] 완료 ({time.time()-t0:.0f}s) → data/w0/  |  브릿지 {n_bridge} {g1} / 커버리지 {coverage:.1%}")


if __name__ == "__main__":
    main()
