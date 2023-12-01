import os

from langchain.indexes import GraphIndexCreator
from langchain.indexes.graph import NetworkxEntityGraph
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.chains import GraphQAChain

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY
os.environ['SERPER_API_KEY'] = c.SERPER_API_KEY
os.environ['GOOGLE_API_KEY'] = c.GOOGLE_API_KEY
os.environ['GOOGLE_CSE_ID'] = c.GOOGLE_CSE_ID


if __name__ == '__main__':
    # Create the graph
    index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))

    with open('./state_of_the_union.txt') as f:
        all_text = f.read()

    text = '\n'.join(all_text.split('\n\n')[105:108])
    print(text)

    graph = index_creator.from_text(text)
    print(graph.get_triples())

    # Querying the graph
    chain = GraphQAChain.from_llm(OpenAI(temperature=0), graph=graph, verbose=True)

    print(chain.run('What is Intel goint to build?'))

    # Save the graph
    graph.write_to_gml("graph.gml")

    # Load the graph
    loaded_graph = NetworkxEntityGraph.from_gml('graph.gml')
    print(loaded_graph.get_triples())
