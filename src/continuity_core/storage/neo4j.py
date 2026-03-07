from __future__ import annotations

from typing import Dict, List, Optional

from neo4j import GraphDatabase

from continuity_core.graph.canonical_schema import GraphEdge, GraphNode, NodeType


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def upsert_nodes(self, nodes: List[GraphNode]) -> int:
        if not nodes:
            return 0
        rows = [n.to_dict() for n in nodes]
        query = """
        UNWIND $rows AS row
        MERGE (n:PKMNode {id: row.id})
        SET n += row.props
        """
        with self._driver.session() as session:
            session.run(query, rows=rows)
        return len(rows)

    def upsert_edges(self, edges: List[GraphEdge]) -> int:
        if not edges:
            return 0
        rows = [e.to_dict() for e in edges]
        query = """
        UNWIND $rows AS row
        MERGE (s:PKMNode {id: row.start_id})
        SET s.node_type = coalesce(row.start_node_type, s.node_type)
        MERGE (t:PKMNode {id: row.end_id})
        SET t.node_type = coalesce(row.end_node_type, t.node_type)
        MERGE (s)-[r:PKM_REL {rel_type: row.rel_type}]->(t)
        SET r += row.props
        """
        with self._driver.session() as session:
            session.run(query, rows=rows)
        return len(rows)

    # -- Resolution graph operations ------------------------------------

    def supersede_node(
        self,
        old_text: str,
        new_text: str,
        reason: str = "",
    ) -> bool:
        """Mark a node as superseded by a newer version.

        Creates a SUPERSEDES edge from the new node to the old one and
        reduces the old node's confidence to 30%.
        """
        query = """
        MATCH (old:PKMNode)
        WHERE old.name = $old_text OR old.description = $old_text
        WITH old LIMIT 1
        OPTIONAL MATCH (new:PKMNode)
        WHERE new.name = $new_text OR new.description = $new_text
        WITH old, new LIMIT 1
        SET old.confidence = 0.3, old.superseded = true
        WITH old, new
        WHERE new IS NOT NULL
        MERGE (new)-[r:SUPERSEDES]->(old)
        SET r.reason = $reason, r.created_at = datetime({timezone:'UTC'})
        RETURN old.id AS old_id
        """
        try:
            with self._driver.session() as session:
                result = session.run(
                    query,
                    old_text=old_text,
                    new_text=new_text,
                    reason=reason,
                )
                return result.single() is not None
        except Exception:
            return False

    def weaken_edge(
        self,
        start_text: str,
        end_text: str,
        rel_type: str,
        factor: float,
    ) -> bool:
        """Multiply an edge's confidence by *factor* (0–1)."""
        query = """
        MATCH (s:PKMNode)-[r:PKM_REL {rel_type: $rel_type}]->(t:PKMNode)
        WHERE (s.name = $start_text OR s.description = $start_text)
          AND (t.name = $end_text OR t.description = $end_text)
        SET r.confidence = coalesce(r.confidence, 1.0) * $factor
        RETURN r.confidence AS new_conf
        """
        try:
            with self._driver.session() as session:
                result = session.run(
                    query,
                    start_text=start_text,
                    end_text=end_text,
                    rel_type=rel_type,
                    factor=factor,
                )
                return result.single() is not None
        except Exception:
            return False

    def bridge_clusters(
        self,
        node_a_text: str,
        node_b_text: str,
        evidence: str = "",
        confidence: float = 0.5,
    ) -> bool:
        """Create a RELATED_TO edge bridging two weakly connected clusters."""
        query = """
        MATCH (a:PKMNode)
        WHERE a.name = $a_text OR a.description = $a_text
        WITH a LIMIT 1
        MATCH (b:PKMNode)
        WHERE b.name = $b_text OR b.description = $b_text
        WITH a, b LIMIT 1
        MERGE (a)-[r:PKM_REL {rel_type: 'RELATED_TO'}]->(b)
        SET r.evidence = $evidence,
            r.confidence = $confidence,
            r.created_at = datetime({timezone:'UTC'})
        RETURN a.id AS a_id, b.id AS b_id
        """
        try:
            with self._driver.session() as session:
                result = session.run(
                    query,
                    a_text=node_a_text,
                    b_text=node_b_text,
                    evidence=evidence,
                    confidence=confidence,
                )
                return result.single() is not None
        except Exception:
            return False

    # -- Query -----------------------------------------------------------

    def query_nodes(
        self, text: Optional[str], node_types: Optional[List[NodeType]], limit: int = 20
    ) -> List[Dict]:
        query = "MATCH (n:PKMNode) WHERE 1=1"
        params: Dict[str, object] = {"limit": limit}
        if text:
            query += (
                " AND (toLower(n.name) CONTAINS toLower($text) "
                "OR toLower(coalesce(n.description,'')) CONTAINS toLower($text))"
            )
            params["text"] = text
        if node_types:
            query += " AND n.node_type IN $types"
            params["types"] = [t.value for t in node_types]
        query += (
            " RETURN n, size((n)--()) AS degree"
            " ORDER BY coalesce(n.updated_at, datetime({timezone:'UTC'})) DESC"
            " LIMIT $limit"
        )
        with self._driver.session() as session:
            res = session.run(query, **params)
            out: List[Dict] = []
            for r in res:
                node = dict(r["n"])
                node["degree"] = r["degree"]
                out.append(node)
            return out
