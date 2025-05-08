from typing import Dict, List, Any
from typing_extensions import TypedDict
from langchain_core.runnables import RunnablePassthrough, chain
from langgraph.graph import StateGraph
from neo4j.exceptions import CypherSyntaxError, ClientError
from llm_agent import LLMAgent
from graph_connection import Neo4jConnection
from graph_schema_extractor import (
    get_concept_nodes,
    get_instance_names_by_concept,
    get_concept_nodes_with_relationships,
    get_one_instance_per_concept_by_name,
    get_graph_schema,
)
from query_reviewer import review_query_agent
from results_explainer import explain_results_agent

MAX_RETRIES = 3

class GraphState(TypedDict):
    """
    Representa el estado de nuestro grafo de búsqueda.
    """
    user_question: str
    concept_nodes: List[str]
    concept_instance_names: Dict[str, List[str]]
    concept_relationships: List[Dict[str, Any]]
    one_instance_per_concept: Dict[str, Dict[str, Any]]
    graph_schema: List[Dict[str, Any]]
    cypher_query: str = None
    query_result: List[Dict[str, Any]] = None
    error: str = None
    retry_count: int = 0
    final_answer: str = None
    llm_agent: LLMAgent
    graph_connection: Neo4jConnection

def generate_query(state: GraphState):
    return {"cypher_query": state["llm_agent"].generate_cypher_query(
        user_question=state["user_question"],
        concept_nodes=state["concept_nodes"],
        concept_instance_names=state["concept_instance_names"],
        concept_relationships=state["concept_relationships"],
        one_instance_per_concept=state["one_instance_per_concept"],
        graph_schema=state["graph_schema"]
    )}

def execute_query(state: GraphState):
    query = state.get("cypher_query")
    graph_connection = state.get("graph_connection")
    try:
        results = graph_connection.execute_and_fetch(query)
        return {"query_result": results, "error": None}
    except CypherSyntaxError as e:
        return {"query_result": None, "error": str(e)}
    except ClientError as e:
        return {"query_result": None, "error": str(e)}
    except Exception as e:
        return {"query_result": None, "error": f"Error al ejecutar la consulta: {e}"}

def review_query(state: GraphState):
    return review_query_agent.invoke(state)

def explain_results(state: GraphState):
    return explain_results_agent.invoke(state)

def check_query_error_condition(state: GraphState):
    return "review" if state.get("error") else "explain"

def check_execution_result_condition(state: GraphState):
    if state.get("error"):
        return {"__next__": "increment_retry_count"}
    else:
        return {"__next__": "explain_results"}

def should_retry_condition(state: GraphState):
    return "retry" if state.get("retry_count", 0) < MAX_RETRIES and state.get("error") else "respond"

def increment_retry_count(state: GraphState):
    return {"retry_count": state.get("retry_count", 0) + 1}

def final_response(state: GraphState):
    return {"final_answer": state.get("final_answer", "No se pudo obtener una respuesta.")}

def create_langgraph_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("generate_query", generate_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("check_execution_result", check_execution_result_condition)
    workflow.add_node("review_query", review_query)
    workflow.add_node("explain_results", explain_results)
    workflow.add_node("final_response", final_response)
    workflow.add_node("increment_retry_count", increment_retry_count)

    workflow.set_entry_point("generate_query")

    workflow.add_edge("generate_query", "execute_query")
    workflow.add_edge("execute_query", "check_execution_result")

    workflow.add_conditional_edges(
        "check_execution_result",
        lambda state: "error" if state.get("error") else "success",
        {
            "success": "explain_results",
            "error": "increment_retry_count",
        },
    )

    workflow.add_edge("increment_retry_count", "generate_query") # Volver a generar la query

    # Aristas condicionales para el reintento (desde execute_query)
    workflow.add_conditional_edges(
        "execute_query",
        lambda state: "retry" if state.get("retry_count", 0) < MAX_RETRIES and state.get("error") else "final_response",
        {
            "retry": "increment_retry_count",
            "final_response": "final_response", # Directamente a final_response si no retry
        },
    )

    workflow.add_edge("explain_results", "final_response")

    app = workflow.compile()

    return app

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

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
        workflow = create_langgraph_workflow()

        def run_search(question: str):
            concept_nodes_data = get_concept_nodes(graph_connection)
            concept_names = [node['n'].get('name') for node in concept_nodes_data if node.get('n') and node['n'].get('name')]
            concept_instance_names = get_instance_names_by_concept(graph_connection)
            concept_relationships = get_concept_nodes_with_relationships(graph_connection)
            one_instance_per_concept = get_one_instance_per_concept_by_name(graph_connection)
            graph_schema = get_graph_schema(graph_connection)

            inputs = {
                "user_question": question,
                "concept_nodes": concept_names,
                "concept_instance_names": concept_instance_names,
                "concept_relationships": concept_relationships,
                "one_instance_per_concept": one_instance_per_concept,
                "graph_schema": graph_schema,
                "llm_agent": llm_agent,
                "graph_connection": graph_connection,
            }
            output = workflow.invoke(inputs)
            return output.get("final_answer", "No se pudo obtener una respuesta.")

        while True:
            pregunta_usuario = input("\nIngresa tu pregunta (o escribe 'salir' para terminar): ")
            if pregunta_usuario.lower() == 'salir':
                break

            respuesta = run_search(pregunta_usuario)
            print("\nRespuesta:")
            print(respuesta)