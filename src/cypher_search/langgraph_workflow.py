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
    print(f"\n🔍 Consulta Cypher generada: {query}")
    graph_connection = state.get("graph_connection")
    try:
        results = graph_connection.execute_and_fetch(query)
        # print('Results:', results)
        return {"query_result": results, "error": None}
    except CypherSyntaxError as e:
        return {"query_result": None, "error": str(e)}
    except ClientError as e:
        return {"query_result": None, "error": str(e)}
    except Exception as e:
        return {"query_result": None, "error": f"Error al ejecutar la consulta: {e}"}

def explain_results(state: GraphState):
    # Si hay resultados, usar el agente explicador, sino devolver mensaje por defecto
    if state.get("query_result"):
        result = explain_results_agent.invoke(state)
        return {"final_answer": result.get("final_answer", "Resultados procesados correctamente.")}
    else:
        return {"final_answer": "No se encontraron resultados para la consulta."}

def increment_retry_count(state: GraphState):
    return {"retry_count": state.get("retry_count", 0) + 1}

def final_response(state: GraphState):
    return {"final_answer": state.get("final_answer", "No se pudo obtener una respuesta.")}

def create_langgraph_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("generate_query", generate_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("explain_results", explain_results)
    workflow.add_node("final_response", final_response)
    workflow.add_node("increment_retry_count", increment_retry_count)

    workflow.set_entry_point("generate_query")

    # Flujo principal
    workflow.add_edge("generate_query", "execute_query")
    
    # Después de ejecutar query, decidir si hay error o éxito
    workflow.add_conditional_edges(
        "execute_query",
        lambda state: "error" if state.get("error") and state.get("retry_count", 0) < MAX_RETRIES else "success",
        {
            "error": "increment_retry_count",
            "success": "explain_results",
        },
    )

    # Después de incrementar retry, volver a generar query
    workflow.add_edge("increment_retry_count", "generate_query")
    
    # Después de explicar resultados, ir a respuesta final
    workflow.add_edge("explain_results", "final_response")

    app = workflow.compile()

    return app

if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

    # Cargar .env desde la raíz del proyecto
    load_dotenv(dotenv_path='../../.env')
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    print(f"Credenciales Neo4j:")
    print(f"  URI: {uri}")
    print(f"  User: {user}")
    print(f"  Password: {'*' * len(password) if password else 'None'}")
    print(f"  OpenAI API Key: {'*' * 10 + openai_api_key[-4:] if openai_api_key else 'None'}")

    if not all([uri, user, password, openai_api_key]):
        # print("Por favor, configura las variables de entorno NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD y OPENAI_API_KEY en tu .env file.")
        pass
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
            
            # Retornar información completa
            return {
                "answer": output.get("final_answer", "No se pudo obtener una respuesta."),
                "cypher_query": output.get("cypher_query", "No se generó consulta"),
                "query_result": output.get("query_result", []),
                "error": output.get("error", None)
            }

        while True:
            pregunta_usuario = input("\nIngresa tu pregunta (o escribe 'salir' para terminar): ")
            if pregunta_usuario.lower() == 'salir':
                break

            resultado = run_search(pregunta_usuario)
            
            print("\n" + "="*60)
            print("📊 RESULTADOS DE LA BÚSQUEDA")
            print("="*60)
            
            if resultado["error"]:
                print(f"❌ Error: {resultado['error']}")
            
            print(f"\n📝 Consulta Cypher:")
            print(f"   {resultado['cypher_query']}")
            
            print(f"\n📈 Resultados encontrados: {len(resultado['query_result'])}")
            
            print(f"\n💬 Respuesta del agente:")
            print(f"   {resultado['answer']}")
            
            if resultado['query_result'] and len(resultado['query_result']) <= 5:
                print(f"\n🔍 Datos brutos (primeros resultados):")
                for i, item in enumerate(resultado['query_result'][:5], 1):
                    print(f"   {i}. {item}")
