import typer
from agent.query_agent import QueryAgent

app = typer.Typer()

@app.command()
def ask(question: str):
    """Haz una pregunta al agente LLM."""
    agent = QueryAgent()
    result = agent.ask(question)
    print("Respuesta del agente:\n")
    print(result)

@app.command("ask-interactive")
def ask_interactive():
    """Sesión interactiva para hacer múltiples preguntas al agente."""
    agent = QueryAgent()
    print("Sesión interactiva iniciada. Escribe 'salir' para terminar.\n")
    while True:
        question = input("Tu pregunta: ")
        if question.lower() in {"salir", "exit", "q"}:
            break
        result = agent.ask(question)
        print("Respuesta:\n", result)

if __name__ == "__main__":
    app()