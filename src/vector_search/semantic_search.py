import os
import json
import concurrent.futures
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

def evaluate_relevance_with_llm(question: str, node: Dict[str, Any], relation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evalúa la relevancia de un nodo y su relación con respecto a la pregunta utilizando un LLM.

    Args:
        question (str): La pregunta original del usuario.
        node (Dict[str, Any]): La información del nodo a evaluar.
        relation (Dict[str, Any]): La información de la relación que conecta al nodo.

    Returns:
        Dict[str, Any]: Un diccionario con la puntuación de relevancia y la justificación del LLM.
    """
    try:
        content = f"""Determina si el siguiente nodo y su relación son relevantes para la pregunta: '{question}'.

        Información del Nodo:
        ID: {node.get("id")}
        Labels: {node.get("labels")}
        Propiedades: {node.get("properties")}

        Información de la Relación:
        Tipo: {relation.get("type")}
        Propiedades: {relation.get("properties")}

        Proporciona una puntuación de relevancia (0.0 a 1.0) y una breve justificación de tu decisión.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": content}
            ],
            response_format={"type": "json_object"}
        )
        json_response = response.choices[0].message.content
        if json_response:
            return json.loads(json_response)
        else:
            return {"relevance_score": 0.0, "justification": "No se pudo obtener una respuesta del LLM."}
    except Exception as e:
        print(f"Error al contactar al LLM: {e}")
        return {"relevance_score": 0.0, "justification": f"Error al contactar al LLM: {e}"}

def embed_node_property(text: str):
    """Función para empaquetar la generación de embeddings."""
    return embed_text(text)

def evaluate_relation_parallel(question: str, end_node: Dict[str, Any], relation: Dict[str, Any]):
    """Función para empaquetar la evaluación de relevancia."""
    return evaluate_relevance_with_llm(question, end_node, relation)

def _semantic_search_with_precomputed_embeddings(
    neo4j: Neo4jConnection,
    question: str,
    top_k: int,
    depth: int,
    max_workers: int
) -> Dict[str, Any]:
    """Realiza la búsqueda semántica utilizando embeddings precalculados."""
    print("Utilizando búsqueda semántica con embeddings precalculados.")
    query_embedding = embed_text(question)
    nodes_with_embeddings = get_all_nodes_with_embeddings(neo4j)
    scored_nodes = []
    for node in nodes_with_embeddings:
        best_score = 0.0
        for key, value in node.items():
            if key.endswith("_embedding") and isinstance(value, list):
                score = cosine_similarity(query_embedding, value)
                if score > best_score:
                    best_score = score
        scored_nodes.append((node, best_score))

    top_nodes_with_score = sorted(scored_nodes, key=lambda x: x[1], reverse=True)[:top_k]
    top_nodes = [item[0] for item in top_nodes_with_score]
    initial_nodes_info = [{"node": item[0], "relevance_score": item[1]} for item in top_nodes_with_score]
    top_node_ids = [n.get("id") for n in top_nodes]

    expanded_relationships = expand_with_relationships(neo4j, top_node_ids, depth=depth)
    evaluated_context = _parallel_evaluate_relationships(question, expanded_relationships, max_workers)

    return {
        "question": question,
        "initial_nodes": initial_nodes_info,
        "context": evaluated_context,
        "depth": depth
    }

def _semantic_search_dynamic_embeddings_parallel(
    neo4j: Neo4jConnection,
    question: str,
    top_k: int,
    depth: int,
    max_workers: int
) -> Dict[str, Any]:
    """Realiza la búsqueda semántica generando embeddings dinámicamente de forma paralela."""
    print("Realizando búsqueda semántica generando embeddings dinámicamente para todas las propiedades de texto de forma paralela.")
    query_embedding = embed_text(question)
    query_all_nodes = """
    MATCH (n)
    RETURN n
    """
    all_nodes = [record["n"] for record in neo4j.execute_and_fetch(query_all_nodes)]

    scored_nodes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for node in all_nodes:
            node_id = node.get("id")
            for key, value in node.items():
                if isinstance(value, str):
                    future = executor.submit(embed_node_property, value)
                    futures.append((node, key, future))

        node_best_scores = {}
        node_embedding_sources = {}
        for node, key, future in futures:
            try:
                embedding = future.result()
                score = cosine_similarity(query_embedding, embedding)
                node_id = node.get("id")
                if node_id not in node_best_scores or score > node_best_scores[node_id]:
                    node_best_scores[node_id] = score
                    node_embedding_sources[node_id] = f"key: {key}"
            except Exception as e:
                print(f"Error embedding property for node {node.get('id')}: {e}")

    for node in all_nodes:
        node_id = node.get("id")
        if node_id in node_best_scores and node_best_scores[node_id] > -1.0:
            scored_nodes.append((node, node_best_scores[node_id], node_embedding_sources[node_id]))

    top_nodes_with_score_source = sorted(scored_nodes, key=lambda x: x[1], reverse=True)[:top_k]
    top_nodes = [item[0] for item in top_nodes_with_score_source]
    initial_nodes_info = [{"node": item[0], "relevance_score": item[1], "embedding_source": item[2]} for item in top_nodes_with_score_source]
    top_node_ids = [n.get("id") for n in top_nodes]

    expanded_relationships = expand_with_relationships(neo4j, top_node_ids, depth=depth)
    evaluated_context = _parallel_evaluate_relationships(question, expanded_relationships, max_workers)

    return {
        "question": question,
        "initial_nodes": initial_nodes_info,
        "context": evaluated_context,
        "depth": depth
    }

def _parallel_evaluate_relationships(question: str, expanded_relationships: Dict[str, List[Dict[str, Any]]], max_workers: int) -> Dict[str, List[Dict[str, Any]]]:
    """Evalúa las relaciones expandidas en paralelo."""
    evaluated_context = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_relation_parallel, question, rel_info["end_node"], rel_info["relation"])
                   for start_node_id, relations in expanded_relationships.items()
                   for rel_info in relations]
        results = [future.result() for future in futures]

    index = 0
    for start_node_id, relations in expanded_relationships.items():
        evaluated_relations = []
        for rel_info in relations:
            if index < len(results):
                evaluation = results[index]
                evaluated_relations.append({
                    "relation": rel_info["relation"],
                    "end_node": rel_info["end_node"],
                    "relevance": evaluation.get("relevance_score"),
                    "justification": evaluation.get("justification")
                })
                index += 1
        evaluated_context[start_node_id] = evaluated_relations
    return evaluated_context

def semantic_search_with_context_parallel(
    neo4j: Neo4jConnection,
    question: str,
    top_k: int = 3,
    depth: int = 3,
    max_workers: int = 5
) -> Dict[str, Any]:
    """
    Realiza una búsqueda semántica paralelizada. Primero verifica si existen campos de embedding
    (con el sufijo '_embedding') en los nodos y utiliza la función correspondiente.

    Args:
        neo4j (Neo4jConnection): Conexión a Neo4j.
        question (str): Pregunta del usuario.
        top_k (int, optional): Número máximo de nodos similares a recuperar. Por defecto 3.
        depth (int, optional): La profundidad máxima para expandir las relaciones. Por defecto 1.
        max_workers (int, optional): Número máximo de threads a usar para la paralelización. Por defecto 5.

    Returns:
        Dict[str, Any]: Diccionario con los nodos relevantes y su contexto relacional evaluado por el LLM.
    """
    nodes_with_existing_embeddings = get_all_nodes_with_embeddings(neo4j)

    if nodes_with_existing_embeddings:
        return _semantic_search_with_precomputed_embeddings(neo4j, question, top_k, depth, max_workers)
    else:
        return _semantic_search_dynamic_embeddings_parallel(neo4j, question, top_k, depth, max_workers)

def expand_with_relationships(neo4j: Neo4jConnection, start_node_ids: List[str], depth: int = 3, current_depth: int = 0, visited: set = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Recupera nodos y relaciones adyacentes para un conjunto de nodos dado, hasta una profundidad específica.

    Args:
        neo4j (Neo4jConnection): Instancia conectada a la base de datos Neo4j.
        start_node_ids (List[str]): Lista de IDs de nodos desde los cuales comenzar la expansión.
        depth (int, optional): La profundidad máxima de la búsqueda. Por defecto 1.
        current_depth (int, optional): La profundidad actual de la búsqueda (para uso interno en la recursión). Por defecto 0.
        visited (set, optional): Conjunto de IDs de nodos ya visitados para evitar ciclos. Por defecto None.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Diccionario donde las claves son los IDs de los nodos de inicio
                                         y los valores son listas de diccionarios con nodos (n, m) y relaciones (r) encontradas.
    """
    if visited is None:
        visited = set(start_node_ids)

    if current_depth >= depth or not start_node_ids:
        return {}

    results = {}
    next_level_nodes = set()

    for start_node_id in start_node_ids:
        query = f'''
        MATCH (n)-[r]-(m)
        WHERE n.id = $start_node_id AND m.id <> $start_node_id // Evitar volver al nodo inicial inmediatamente
        RETURN n, r, m
        '''
        records = neo4j.execute_and_fetch(query, {"start_node_id": start_node_id})
        relationships = []
        for record in records:
            n_data = record.get("n")
            r_data = record.get("r")
            m_data = record.get("m")
            relationships.append({
                "start_node": {"id": n_data.get("id"), "labels": n_data.get("labels"), "properties": n_data.get("properties")},
                "relation": r_data,
                "end_node": {"id": m_data.get("id"), "labels": m_data.get("labels"), "properties": m_data.get("properties")}
            })
            if m_data.get("id") not in visited:
                next_level_nodes.add(m_data.get("id"))
                visited.add(m_data.get("id"))
        results[start_node_id] = relationships

    next_level_results = expand_with_relationships(neo4j, list(next_level_nodes), depth, current_depth + 1, visited)

    for node_id, next_level_rels in next_level_results.items():
        if node_id in results:
            results[node_id].extend(next_level_rels)
        else:
            results[node_id] = next_level_rels

    return results
