import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List, Any
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLMAgent:
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_cypher_query(self, user_question: str, concept_nodes: List[str], concept_instance_names: Dict[str, List[str]], concept_relationships: List[Dict[str, Any]], one_instance_per_concept: Dict[str, Dict[str, Any]], graph_schema: List[Dict[str, Any]]) -> str:
        """
        Generates a Cypher query based on the user question and the provided graph context.
        Removes any markdown code block formatting from the LLM's response.
        """
        formatted_concept_relationships = []
        seen_relationships = set()
        for relation_data in concept_relationships:
            c1_name = relation_data.get('c1', {}).get('name')
            r_value = relation_data.get('r')
            r_type = r_value[1] if isinstance(r_value, tuple) and len(r_value) > 1 else None
            c2_name = relation_data.get('c2', {}).get('name')
            if c1_name and r_type and c2_name:
                relationship_tuple = (c1_name, r_type, c2_name)
                if relationship_tuple not in seen_relationships:
                    formatted_concept_relationships.append(f"{c1_name} -[{r_type}]-> {c2_name}")
                    seen_relationships.add(relationship_tuple)

        prompt = f"""Eres un agente experto en traducir preguntas en lenguaje natural a consultas Cypher para una base de datos de grafos Neo4j. Es crucial que las consultas generadas utilicen las etiquetas de nodo, los tipos de relaciones y los nombres de propiedades que se encuentran en el siguiente contexto.
        **TIENES QUE USAR LOS NODOS Y RELACIONES QUE APARECEN EN EL CONTEXTO. NO PUEDES HACER SUPOSICIONES NI USAR NOMBRES QUE NO APARECEN EN EL CONTEXTO.**

        Contexto de la Base de Datos:

        Nodos Concepto: Estos son los tipos principales de entidades en la base de datos:
        {concept_nodes}

        Ejemplos de nombres de instancias por cada Nodo Concepto (para entender el vocabulario).
        {concept_instance_names}

        Relaciones entre Nodos Concepto: Esto describe cómo se relacionan los tipos principales de entidades entre sí:
        {formatted_concept_relationships}

        Ejemplo de una instancia por cada Nodo Concepto (para entender las propiedades). **Con esto puedes saber como escribir las consultas en cypher respecto a los campos de cada nodo. Es super importante**::
        {one_instance_per_concept}

        Esquema de Relaciones del Grafo: Esto muestra los tipos de relaciones posibles y las etiquetas de los nodos que conectan.
        {graph_schema}

        Pregunta del Usuario: "{user_question}"

        Basándote estrictamente en el contexto proporcionado, genera una consulta Cypher que responda a la pregunta del usuario. La consulta debe ser lo más precisa y eficiente posible para obtener la información necesaria. Utiliza solo las etiquetas de nodo, los tipos de relaciones y los nombres de propiedades que aparecen en el contexto.

        Solo devuelve la consulta Cypher. No incluyas explicaciones ni texto adicional.
        """
        print(prompt)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            cypher_query = response.choices[0].message.content.strip()
            match = re.match(r"```(?:cypher)?\n(.*?)\n```", cypher_query, re.DOTALL)
            if match:
                return match.group(1)
            return cypher_query
        except Exception as e:
            print(f"Error al contactar al LLM: {e}")
            return ""