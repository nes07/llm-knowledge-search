from typing import List, Dict, Any
from graph_connection import Neo4jConnection
from concept_info_agent import ConceptInfoGeneratorAgent
import os
from dotenv import load_dotenv

def get_all_node_labels(graph_connection: Neo4jConnection) -> List[str]:
    """
    Obtiene todas las etiquetas de nodo únicas de la base de datos Memgraph.
    """
    query = """
    MATCH (n)
    UNWIND labels(n) AS label
    RETURN DISTINCT label
    """
    result = graph_connection.execute_and_fetch(query)
    return [label['label'] for label in result if label['label']] if result else []

def get_all_relationship_types(graph_connection: Neo4jConnection) -> List[str]:
    """
    Obtiene todos los tipos de relaciones únicos de la base de datos Memgraph.
    """
    query = """
    MATCH ()-[r]->()
    RETURN DISTINCT type(r) AS relationshipType
    """
    result = graph_connection.execute_and_fetch(query)
    return [row['relationshipType'] for row in result if row['relationshipType']] if result else []

def get_node_properties_by_label(graph_connection: Neo4jConnection, label: str, limit: int = 5) -> List[str]:
    """
    Obtiene una muestra de las propiedades para nodos con una etiqueta dada.
    """
    query = f"""
    MATCH (n:`{label}`)
    WITH keys(n) AS properties
    UNWIND properties AS prop
    RETURN DISTINCT prop
    LIMIT $limit
    """
    result = graph_connection.execute_and_fetch(query, {"label": label, "limit": limit})
    return [row['prop'] for row in result] if result else []

def get_detailed_graph_schema(graph_connection: Neo4jConnection) -> List[Dict[str, Any]]:
    """
    Obtiene un esquema detallado del grafo (etiquetas y tipos de relaciones).
    """
    query = """
    MATCH (n)-[r]->(m)
    RETURN DISTINCT labels(n) AS source_labels, type(r) AS relationship_type, labels(m) AS target_labels
    """
    return graph_connection.execute_and_fetch(query)

def create_concept_node(graph_connection: Neo4jConnection, name: str, information: str) -> None:
    """
    Crea un nodo Concept en la base de datos con la etiqueta 'Concept' y una etiqueta adicional basada en su nombre.
    """
    query = f"""
    CREATE (c:Concept:`{name}` {{name: $name, information: $information}})
    """
    graph_connection.execute_and_fetch(query, {"name": name, "information": information})
    print(f"Nodo Concept creado para: {name} con etiquetas :Concept y :{name}")

def create_concept_relationship(graph_connection: Neo4jConnection, source_concept_name: str, relationship_type: str, target_concept_name: str) -> None:
    """
    Crea una relación entre dos nodos Concept basándose en sus nombres y el tipo de relación.
    """
    query = f"""
    MATCH (source:Concept {{name: $source_concept_name}}), (target:Concept {{name: $target_concept_name}})
    CREATE (source)-[:`{relationship_type}`]->(target)
    """
    graph_connection.execute_and_fetch(query, {"source_concept_name": source_concept_name, "target_concept_name": target_concept_name, "relationship_type": relationship_type})
    print(f"Relación Concept creada: {source_concept_name} -[:{relationship_type}]-> {target_concept_name}")

def generate_and_insert_concepts(uri: str, user: str, password: str):
    """
    Función principal que coordina la obtención del esquema, la generación de los nodos concepto (usando el agente) y su inserción,
    e infiere y crea relaciones entre los nodos concepto.
    """
    graph_connection = Neo4jConnection(uri, user, password)
    llm_agent = ConceptInfoGeneratorAgent(api_key=os.getenv("OPENAI_API_KEY"))

    node_labels = get_all_node_labels(graph_connection)
    detailed_schema = get_detailed_graph_schema(graph_connection)
    concept_nodes_map = {}

    for label in node_labels:
        if label == "Concept":
            continue
        properties = get_node_properties_by_label(graph_connection, label)
        related_relationships = [
            rel['relationship_type']
            for rel in detailed_schema
            if label in rel['source_labels'] or label in rel['target_labels']
        ]
        information = llm_agent.generate_information(label, related_relationships, properties, detailed_schema, get_all_relationship_types(graph_connection))
        create_concept_node(graph_connection, label, information)
        concept_nodes_map[label] = label

    for relation in detailed_schema:
        source_labels = relation['source_labels']
        relationship_type = relation['relationship_type']
        target_labels = relation['target_labels']

        if source_labels and target_labels and relationship_type:
            source_concept_name = source_labels[0]
            target_concept_name = target_labels[0]

            if source_concept_name in concept_nodes_map and target_concept_name in concept_nodes_map:
                create_concept_relationship(graph_connection, source_concept_name, relationship_type, target_concept_name)
                print(f"Relación Concept creada: {source_concept_name} -[:{relationship_type}]-> {target_concept_name}")

if __name__ == '__main__':
    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user= os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password, os.getenv("OPENAI_API_KEY")]):
        print("Por favor, configura las variables de entorno NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD y OPENAI_API_KEY en tu .env file.")
    else:
        generate_and_insert_concepts(neo4j_uri, neo4j_user, neo4j_password)