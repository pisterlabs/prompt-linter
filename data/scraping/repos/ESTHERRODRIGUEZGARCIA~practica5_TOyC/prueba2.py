from langchain.llms import OpenAI
from langchain.indexes import GraphIndexCreator


text = "Apple announced the Vision Pro in 2023."

index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
graph = index_creator.from_text(text)
graph.get_triples()
