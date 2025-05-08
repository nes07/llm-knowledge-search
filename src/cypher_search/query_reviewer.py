from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

review_prompt_template = """Eres un agente experto en revisar y corregir consultas Cypher para Neo4j.
La consulta original fue:
{cypher_query}
La pregunta del usuario fue:
{user_question}
Neo4j devolvió la siguiente excepción:
{error}
Basándote en la excepción, revisa y corrige la consulta Cypher para que sea válida y responda a la pregunta del usuario.
Solo devuelve la consulta Cypher corregida. No incluyas explicaciones.
"""
review_prompt = PromptTemplate.from_template(review_prompt_template)
review_query_agent = review_prompt | ChatOpenAI(model_name="gpt-4-turbo-preview") | {"revised_query": lambda x: x.content}