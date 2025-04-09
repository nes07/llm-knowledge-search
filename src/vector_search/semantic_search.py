import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def embed_text(text: str) -> List[float]:
    """
    Convierte un texto en un vector de embeddings usando el modelo text-embedding-3-large de OpenAI.

    Args:
        text (str): El texto del usuario.

    Returns:
        List[float]: Lista de floats que representa el embedding del texto.
    """
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula la similitud del coseno entre dos vectores usando scikit-learn.

    Args:
        vec1 (List[float]): Primer vector.
        vec2 (List[float]): Segundo vector.

    Returns:
        float: Valor de similitud (1 = idéntico, 0 = ortogonal).
    """
    return float(sk_cosine_similarity([vec1], [vec2])[0][0])

def get_all_nodes_with_embeddings(neo4j: Neo4jConnection) -> List[Dict[str, Any]]:
    """
    Recupera todos los nodos que contienen campos de embedding.

    Args:
        neo4j (Neo4jConnection): Conexión a la base de datos Neo4j.

    Returns:
        List[Dict[str, Any]]: Lista de nodos y sus propiedades.
    """
    query = """
    MATCH (n)
    WHERE any(key IN keys(n) WHERE key ENDS WITH '_embedding')
    RETURN n
    """
    return [record["n"] for record in neo4j.execute_and_fetch(query)]

def semantic_search_with_context(
    neo4j: Neo4jConnection,
    question: str,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Realiza una búsqueda semántica sin usar índice vectorial, comparando embeddings almacenados en los nodos.

    Args:
        neo4j (Neo4jConnection): Conexión a Neo4j.
        question (str): Pregunta del usuario.
        top_k (int, optional): Número de nodos similares a recuperar. Por defecto 3.

    Returns:
        Dict[str, Any]: Diccionario con los nodos relevantes y su contexto relacional.
    """
    query_embedding = embed_text(question)
    nodes = get_all_nodes_with_embeddings(neo4j)

    scored_nodes = []
    for node in nodes:
        best_score = 0.0
        for key, value in node.items():
            if key.endswith("_embedding") and isinstance(value, list):
                score = cosine_similarity(query_embedding, value)
                if score > best_score:
                    best_score = score
        scored_nodes.append((node, best_score))

    top_nodes = sorted(scored_nodes, key=lambda x: x[1], reverse=True)
    seen_ids = set()
    unique_top_nodes = []
    for node, score in top_nodes:
        node_id = node.get("id")
        if node_id and node_id not in seen_ids:
            seen_ids.add(node_id)
            unique_top_nodes.append(node)
        if len(unique_top_nodes) == top_k:
            break

    node_ids = [n["id"] for n in unique_top_nodes]
    context = expand_with_relationships(neo4j, node_ids)

    return {
        "question": question,
        "similar_nodes": unique_top_nodes,
        "expanded_context": context
    }

def expand_with_relationships(neo4j: Neo4jConnection, node_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Recupera nodos y relaciones adyacentes para un conjunto de nodos dado.

    Args:
        neo4j (Neo4jConnection): Instancia conectada a la base de datos Neo4j.
        node_ids (List[str]): Lista de IDs de nodos desde los cuales expandir relaciones.

    Returns:
        List[Dict[str, Any]]: Lista de diccionarios con nodos (n, m) y relaciones (r).
    """
    query = f'''
    MATCH (n)-[r]-(m)
    WHERE n.id IN {node_ids}
    RETURN n, r, m
    '''
    return neo4j.execute_and_fetch(query)
