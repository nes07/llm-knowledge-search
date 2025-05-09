import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ConceptInfoGeneratorAgent:
    def __init__(self, api_key: str, model_name: str = "gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_information(self, label: str, relationships: List[str], properties: List[str], schema: List[Dict[str, Any]], all_relationship_types: List[str]) -> str:
        """
        Generates a concise informative description for a concept node using an LLM, including all relationship types.
        """
        prompt = f"""Eres un experto en comprender esquemas de bases de datos de grafos Neo4j. Tu tarea es generar una breve descripción informativa para un nodo concepto dado su etiqueta, las relaciones en las que participa, sus propiedades típicas y el esquema general del grafo.

        Etiqueta del Nodo: {label}

        Relaciones en las que participa (tipos):
        {relationships}

        Propiedades típicas:
        {properties}

        Esquema general del grafo (ejemplos de relaciones: nodo_origen_labels -> tipo_relacion -> nodo_destino_labels):
        {schema}

        Todos los tipos de relaciones existentes en la base de datos:
        {all_relationship_types}

        Genera una descripción concisa (máximo 2-3 frases) que explique qué representa típicamente un nodo con la etiqueta "{label}" en esta base de datos.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error al contactar al LLM para {label}: {e}")
            return f"Descripción no disponible para {label}."