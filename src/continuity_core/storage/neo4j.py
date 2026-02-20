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

    def query_nodes(self, text: Optional[str], node_types: Optional[List[NodeType]], limit: int = 20) -> List[Dict]:
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
