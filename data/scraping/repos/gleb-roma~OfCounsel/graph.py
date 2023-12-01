from langchain.graphs import Neo4jGraph

url="bolt://localhost:7687"
username="neo4j"
password="ofcounsel"

graph = Neo4jGraph(
    url=url,
    username=username, 
    password=password
)
