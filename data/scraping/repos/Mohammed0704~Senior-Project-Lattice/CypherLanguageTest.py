from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
import sys

#pip3 install neo4j openai langchain #may need to run as Admin

graph = Neo4jGraph(
    url="bolt://localhost:7687", 
    username="neo4j", 
    password="password"
)

import os

os.environ['OPENAI_API_KEY'] = "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"

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True,
)

chain.run("""
What full name of Students where the Student has an outstanding balance as an Integer has more than 97 
are in the College of Computing & Informatics where AreaOfStudy is a part of a Department
""")

sys.exit(0)