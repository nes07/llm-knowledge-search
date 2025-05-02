from typing import List, Dict, Any
from graph_connection import Neo4jConnection

def get_concept_nodes(graph_connection: Neo4jConnection) -> List[Dict[str, Any]]:
    """
    Recupera todos los nodos con la etiqueta 'Concept'.
    """
    query = """
    MATCH (n:Concept)
    RETURN n
    """
    return graph_connection.fetch_all(query)

def get_instance_names_by_concept(graph_connection: Neo4jConnection) -> Dict[str, List[str]]:
    """
    Extrae los nombres de todas las instancias para cada nodo concepto.
    Ahora busca nodos que tienen una relación INSTANCIATED_FROM *hacia* el nodo concepto.
    Assumes instances have a 'name' property.
    """
    concept_nodes = get_concept_nodes(graph_connection)
    instance_names_by_concept = {}
    for concept_node in concept_nodes:
        concept_id = concept_node['n'].get('id')
        if concept_id:
            query = f"""
            MATCH (instance)-[:INSTANCIATED_FROM]->(concept)
            WHERE id(concept) = $concept_id AND exists(instance.name)
            RETURN DISTINCT instance.name AS instance_name
            """
            results = graph_connection.fetch_all(query, {"concept_id": concept_id})
            instance_names_by_concept[concept_id] = [res.get('instance_name') for res in results if res.get('instance_name')]
    return instance_names_by_concept

def get_concept_nodes_with_relationships(graph_connection: Neo4jConnection) -> List[Dict[str, Any]]:
    """
    Recupera los nodos concepto y sus relaciones adyacentes.
    """
    query = """
    MATCH (c1:Concept)-[r]-(c2:Concept)
    RETURN c1, r, c2
    """
    return graph_connection.fetch_all(query)

def get_one_instance_per_concept_by_name(graph_connection: Neo4jConnection) -> Dict[str, Dict[str, Any]]:
    """
    Extrae una instancia por cada nodo concepto para conocer sus campos.
    Ahora busca un nodo que tiene una relación INSTANCIATED_FROM *hacia* el nodo concepto usando el nombre.
    Retorna un diccionario donde la clave es el nombre del concepto y el valor es el diccionario de propiedades de la instancia.
    """
    concept_nodes = get_concept_nodes(graph_connection)
    one_instance_per_concept = {}
    for concept_node in concept_nodes:
        concept_name = concept_node['n'].get('name')
        print(f"Buscando instancia para concepto: {concept_name}")
        if concept_name:
            query = f"""
            MATCH (instance)-[:INSTANCIATED_FROM]->(concept:Concept {{name: $concept_name}})
            RETURN instance
            LIMIT 1
            """
            results = graph_connection.fetch_all(query, {"concept_name": concept_name})
            if results:
                one_instance = results[0].get('instance')
                one_instance_per_concept[concept_name] = one_instance if one_instance else {}
            else:
                one_instance_per_concept[concept_name] = {}
    return one_instance_per_concept

def get_graph_schema(graph_connection: Neo4jConnection) -> List[Dict[str, Any]]:
    """
    Recupera el esquema del grafo (relaciones y las etiquetas de los nodos participantes).
    Adjust the query based on Neo4j's schema retrieval mechanisms.
    """
    query = """
    MATCH (n)-[r]->(m)
    RETURN DISTINCT labels(n) AS source_labels, type(r) AS relationship_type, labels(m) AS target_labels
    """
    return graph_connection.fetch_all(query)