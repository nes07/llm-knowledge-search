from neo4j import GraphDatabase
from typing import List, Dict, Any

class Neo4jConnection:
    """
    Simple wrapper for Neo4j driver session-based execution.
    """
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def execute_and_fetch(self, query: str, parameters: dict = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def fetch_all(self, query: str, parameters: dict = None) -> List[Dict[str, Any]]:
        return self.execute_and_fetch(query, parameters)