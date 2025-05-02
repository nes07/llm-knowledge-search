import os
from dotenv import load_dotenv
from graph_connection import Neo4jConnection
from graph_schema_extractor import (
    get_concept_nodes,
    get_instance_names_by_concept,
    get_concept_nodes_with_relationships,
    get_one_instance_per_concept_by_name,
    get_graph_schema,
)
from llm_agent import LLMAgent
from typing import List, Dict, Any

def semantic_search_using_llm(graph_connection: Neo4jConnection, user_question: str, llm_agent: LLMAgent) -> List[Dict[str, Any]]:
    """
    Realiza una búsqueda semántica traduciendo la pregunta del usuario a una consulta Cypher
    utilizando el contexto del esquema del grafo y los nodos concepto.
    """
    concept_nodes_data = get_concept_nodes(graph_connection)
    concept_names = [node['n'].get('name') for node in concept_nodes_data if node.get('n') and node['n'].get('name')]
    concept_instance_names = get_instance_names_by_concept(graph_connection)
    concept_relationships = get_concept_nodes_with_relationships(graph_connection)
    one_instance_per_concept = get_one_instance_per_concept_by_name(graph_connection)
    graph_schema = get_graph_schema(graph_connection)

    cypher_query = llm_agent.generate_cypher_query(
        user_question=user_question,
        concept_nodes=concept_names,
        concept_instance_names=concept_instance_names,
        concept_relationships=concept_relationships,
        one_instance_per_concept=one_instance_per_concept,
        graph_schema=graph_schema
    )

    if cypher_query:
        print(f"Consulta Cypher generada por el LLM: {cypher_query}")
        results = graph_connection.execute_and_fetch(cypher_query)
        return results
    else:
        print("No se pudo generar una consulta Cypher.")
        return []

if __name__ == '__main__':
    load_dotenv()

    uri = "neo4j://ia-dev.tecnoandina.cl:7687"
    user = "neo4j"
    password = "12345678"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([uri, user, password, openai_api_key]):
        print("Por favor, configura las variables de entorno NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD y OPENAI_API_KEY en tu .env file.")
    else:
        graph_connection = Neo4jConnection(uri, user, password)
        llm_agent = LLMAgent(api_key=openai_api_key)

        while True:
            pregunta_usuario = input("\nIngresa tu pregunta (o escribe 'salir' para terminar): ")
            if pregunta_usuario.lower() == 'salir':
                break

            resultados = semantic_search_using_llm(graph_connection, pregunta_usuario, llm_agent)
            print("\nResultados de la búsqueda:")
            if resultados:
                for resultado in resultados:
                    print(resultado)
            else:
                print("No se encontraron resultados.")

    # if not all([uri, user, password, openai_api_key]):
    #     print("Por favor, configura las variables de entorno NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD y OPENAI_API_KEY en tu .env file.")
    # else:
    #     graph_connection = Neo4jConnection(uri, user, password)
    #     llm_agent = LLMAgent(api_key=openai_api_key)
    #     # pregunta_usuario = "La granja 1 con customer id 2103082000 cuantos pabellones distintos tiene?"
    #     # pregunta_usuario = "Que granja tiene silos?"
    #     pregunta_usuario = "dame los kpi más actuales de los datos de cada crianza para cada customer"

    #     resultados = semantic_search_using_llm(graph_connection, pregunta_usuario, llm_agent)
    #     print("\nResultados de la búsqueda:")
    #     if resultados:
    #         for resultado in resultados:
    #             print(resultado)
    #     else:
    #         print("No se encontraron resultados.")