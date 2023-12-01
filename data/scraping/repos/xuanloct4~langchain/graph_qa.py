import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

from langchain.indexes import GraphIndexCreator
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

index_creator = GraphIndexCreator(llm=llm)

with open("./documents/state_of_the_union.txt") as f:
    all_text = f.read()

text = "\n".join(all_text.split("\n\n")[105:108])
print(text)

graph = index_creator.from_text(text)

graph.get_triples()


from langchain.chains import GraphQAChain
chain = GraphQAChain.from_llm(llm, graph=graph, verbose=True)
chain.run("what is Intel going to build?")


graph.write_to_gml("graph.gml")
from langchain.indexes.graph import NetworkxEntityGraph
loaded_graph = NetworkxEntityGraph.from_gml("graph.gml")
loaded_graph.get_triples()
