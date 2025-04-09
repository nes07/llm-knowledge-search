from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import MemgraphGraph
from langchain_openai import ChatOpenAI
from utils.config import OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MODEL


class QueryAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model=MODEL,_
            temperature=0
        )

        self.graph = MemgraphGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD
        )

        self.chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,
            allow_dangerous_requests=True
        )

    def ask(self, question: str) -> dict:
        print('HERE')
        response = self.chain.invoke(question)
        print(response)
        return response