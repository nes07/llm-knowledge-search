from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any

explain_prompt_template = """La pregunta del usuario fue:
{user_question}
Los resultados de la consulta Cypher son:
{query_result}
Genera una explicación concisa y en lenguaje natural de estos resultados para el usuario.
"""
explain_prompt = PromptTemplate.from_template(explain_prompt_template)
explain_results_agent = explain_prompt | ChatOpenAI(model_name="gpt-4-turbo-preview") | {"final_answer": lambda x: x.content}