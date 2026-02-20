from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class VoidReport:
    void_pairs: List[Tuple[Set[str], Set[str], int]]
    questions: List[str]


class VoidDetector:
    def detect_voids(self, graph: Dict[str, Set[str]], min_size: int = 2,
                     max_pairs: int = 5) -> VoidReport:
        components = self._components(graph)
        components = [c for c in components if len(c) >= min_size]
        voids: List[Tuple[Set[str], Set[str], int]] = []
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                a, b = components[i], components[j]
                edges_between = self._count_edges(graph, a, b)
                if edges_between <= 1:
                    voids.append((a, b, edges_between))
        voids.sort(key=lambda x: (x[2], abs(len(x[0]) - len(x[1]))))
        voids = voids[:max_pairs]
        questions = [self._bridge_question(a, b) for a, b, _ in voids]
        return VoidReport(void_pairs=voids, questions=questions)

    def _components(self, graph: Dict[str, Set[str]]) -> List[Set[str]]:
        seen: Set[str] = set()
        comps: List[Set[str]] = []
        for node in graph.keys():
            if node in seen:
                continue
            comp: Set[str] = set()
            stack = [node]
            seen.add(node)
            while stack:
                cur = stack.pop()
                comp.add(cur)
                for nb in graph.get(cur, set()):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            comps.append(comp)
        return comps

    def _count_edges(self, graph: Dict[str, Set[str]], a: Set[str], b: Set[str]) -> int:
        count = 0
        for u in a:
            for v in graph.get(u, set()):
                if v in b:
                    count += 1
        return count

    def _bridge_question(self, a: Set[str], b: Set[str]) -> str:
        a_label = next(iter(a)) if a else "cluster A"
        b_label = next(iter(b)) if b else "cluster B"
        return (
            f"What evidence or mechanism could connect '{a_label}' and '{b_label}'? "
            "Propose a falsifiable link and a minimal test."
        )
