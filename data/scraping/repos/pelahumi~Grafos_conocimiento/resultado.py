import os
import openai
from langchain.graphs.networkx_graph import KnowledgeTriple
from langchain.llms import OpenAI
from langchain.indexes import GraphIndexCreator
from langchain.chains import GraphQAChain
from langchain.prompts import PromptTemplate
import networkx as nx
import matplotlib.pyplot as plt

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

question = "What team won the 2022 champions league?"
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                          temperature=0, 
                                          messages=[{"role":"user",
                                                    "content":question,}],
                                            max_tokens=1000,)

print(completion["choices"][0]["message"]["content"])

# Knowledge graph
kg = [
    ("Real Madrid", "won", "2022 champions league"),
    ("Chelsea", "won", "2021 champions league"),
    ("Manchester City", "won", "2021 champions league"),
    ("Barsa", "won", "2023 La Liga"),
    ("Real Madrid", "won", "2022 La Liga"),
    ("Atlético de Madrid", "won", "2021 La Liga"),
    ("Sevilla", "won", "2023 europa league"),
    ("Frankfurt", "won", "2022 europa league"),
    ("Villareal", "won", "2021 europa league"),
    ("Manchester City", "won", "2023 premier league"),
    ("Manchester United", "won", "2022 premier league"),
    ("Manchester City", "won", "2021 premier league"),
]

index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))

graph = index_creator.from_text('')
for (node1, relation, node2) in kg:
    graph.add_triple(KnowledgeTriple(node1, relation, node2))

# Create directed graph
G = nx.DiGraph()
for node1, relation, node2 in kg:
    G.add_edge(node1, node2, label=relation)

# Plot the graph
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
