"""GraphPort 구현. NetworkX 인메모리 그래프 (ADR-5).
implements: verihop.ports.GraphPort
edge: relation, source_paragraph_id, evidence · node: type. fwd=out-edge(head), inv=in-edge(tail).
서드파티 import는 이 계층에서만 허용. usecase는 이 파일의 존재를 모른다.
test: usecase 테스트에서 tests/fakes.py로 대체됨
"""
from __future__ import annotations
import pickle
import networkx as nx


class NetworkxGraph:
    def __init__(self, graph_pickle: str):
        with open(graph_pickle, "rb") as f:
            self.g: nx.DiGraph = pickle.load(f)

    def neighbors(self, node, relation, direction):
        if node not in self.g:
            return []
        out: list[tuple[str, list[str]]] = []
        if direction == "fwd":
            for _, t, d in self.g.out_edges(node, data=True):
                if d.get("relation") == relation:
                    out.append((t, [d["source_paragraph_id"]]))
        else:                                        # inv: node가 tail, head를 찾음
            for h, _, d in self.g.in_edges(node, data=True):
                if d.get("relation") == relation:
                    out.append((h, [d["source_paragraph_id"]]))
        return out

    def has_edge(self, src, dst, relation):
        if not self.g.has_edge(src, dst):
            return False
        # DiGraph 단일 엣지: relation 일치 확인 (다중이면 첫 엣지)
        return self.g.get_edge_data(src, dst, {}).get("relation") == relation

    def node_type(self, node):
        return self.g.nodes[node].get("type") if node in self.g else None

    def relations_of(self, node):
        if node not in self.g:
            return []
        rels = {d.get("relation") for _, _, d in self.g.out_edges(node, data=True)}
        rels |= {d.get("relation") for _, _, d in self.g.in_edges(node, data=True)}
        return sorted(r for r in rels if r)

    def exists(self, node):
        return node in self.g
