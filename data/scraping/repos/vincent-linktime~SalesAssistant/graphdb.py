from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
import os

MODEL_NAME = "gpt-3.5-turbo"

class GraphDB:
    def __init__(self):
        self.graph = Neo4jGraph(
            url = os.environ["NEO4J_URL"], 
            username = "neo4j", 
            password = os.environ["NEO4J_PASSWORD"]
        )

        self.graph.refresh_schema()

        self.cypher_chain = GraphCypherQAChain.from_llm(
           cypher_llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME),
           qa_llm = ChatOpenAI(temperature=0), graph=self.graph, verbose=True
        )

    def get_cypher_chain(self):
        return self.cypher_chain
