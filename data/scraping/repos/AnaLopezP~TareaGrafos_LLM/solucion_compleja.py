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

#Grafo complejo
# Conocimientos que le vamos a dar a nuestro grafo
kg = [
    ('Apple', 'is', 'Company'),
    ('Apple', 'created', 'iMac'),
    ('Apple', 'created', 'iPhone'),
    ('Apple', 'created', 'Apple Watch'),
    ('Apple', 'created', 'Vision Pro'),
    ('Apple', 'developed', 'macOS'),
    ('Apple', 'developed', 'iOS'),
    ('Apple', 'developed', 'watchOS'),
    ('Apple', 'is located in', 'USA'),
    ('Steve Jobs', 'co-founded', 'Apple'),
    ('Steve Wozniak', 'co-founded', 'Apple'),
    ('Tim Cook', 'is the CEO of', 'Apple'),
    ('iOS', 'runs on', 'iPhone'),
    ('macOS', 'runs on', 'iMac'),
    ('watchOS', 'runs on', 'Apple Watch'),
    ('Apple', 'was founded in', '1976'),
    ('Apple', 'owns', 'App Store'),
    ('App Store', 'sells', 'iOS apps'),
    ('iPhone', 'announced in', '2007'),
    ('iMac', 'announced in', '1998'),
    ('Apple Watch', 'announced in', '2014'),
    ('Vision Pro', 'announced in', '2023'),
]
#Creamos el grafo
index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
graph = index_creator.from_text(kg)
for (node1, relation, node2) in kg:
    graph.add_triple(KnowledgeTriple(node1, relation, node2))

# Representación gráfica del grafo
G = nx.DiGraph()
for node1, relation, node2 in kg:
    G.add_edge(node1, node2, label=relation)

# Nodo del grafo
plt.figure(figsize=(25, 25), dpi=300)
pos = nx.spring_layout(G, k=2, iterations=50, seed=0)

nx.draw_networkx_nodes(G, pos, node_size=5000)
nx.draw_networkx_edges(G, pos, edge_color='gray', edgelist=G.edges(), width=2)
nx.draw_networkx_labels(G, pos, font_size=12)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

# Display the plot
plt.axis('off')
plt.show()
chain = GraphQAChain.from_llm(OpenAI(temperature=0), graph=graph, verbose=True)
chain.run(question)
