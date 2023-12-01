from langchain.llms import OpenAI
from langchain.indexes import GraphIndexCreator
from langchain.chains import GraphQAChain
from langchain.prompts import PromptTemplate
from langchain.graphs.networkx_graph import KnowledgeTriple
import networkx as nx
import matplotlib.pyplot as plt
import os
import openai

#Pregunta que queremos hacer
os.environ['OPENAI_API_KEY'] = "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"
openai.api_key = os.environ['OPENAI_API_KEY']

question = "When did apple announced the Vision Pro?"
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                            temperature=0, 
                                            messages=[{"role": "user",
                                            "content": question}])

print(completion["choices"][0]["message"]["content"])

#Grafo sencillo
#Creamos un grafo de conocimiento con la siguiente frase
text = "Apple announced the Vision Pro in 2023."

index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
graph = index_creator.from_text(text)
graph.get_triples()

#Representación gráfica del grafo

G = nx.DiGraph() #creamos el grado
G.add_edges_from((source, target, {'relation': relation}) for source, relation, target in graph.get_triples())

# nodos del grafo
plt.figure(figsize=(8,5), dpi=300)
pos = nx.spring_layout(G, k=3, seed=0)

nx.draw_networkx_nodes(G, pos, node_size=2000)
nx.draw_networkx_edges(G, pos, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=12)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.axis('off')
plt.show()
chain = GraphQAChain.from_llm(OpenAI(temperature=0), graph=graph, verbose=True)
chain.run(question)


