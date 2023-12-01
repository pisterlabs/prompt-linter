from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

import logging
import sys
import openai 
import os

def enable_debug():
    openai.log = "debug"

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def get_documents():
    raw_data = "/root/llm_exam/llama_index/examples/paul_graham_essay/data"

    documents = SimpleDirectoryReader(raw_data).load_data()
    return documents


def make_nodes_from_documents(documents):
    parser = SimpleNodeParser.from_defaults()

    nodes = parser.get_nodes_from_documents(documents)
    return nodes


def make_index_from_nodes(nodes):
    from llama_index import VectorStoreIndex

    index = VectorStoreIndex(nodes)
    return index

def make_storage_context(nodes):
    from llama_index import StorageContext

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    return storage_context

def make_indexes_from_storage(storage_context, nodes):
    from llama_index import VectorStoreIndex
    from llama_index import SummaryIndex
    
    index1 = VectorStoreIndex(nodes, storage_context=storage_context)
    index2 = SummaryIndex(nodes, storage_context=storage_context)    
    return index1, index2    


if __name__ == "__main__":
    enable_debug()
    documents = get_documents()
    nodes = make_nodes_from_documents(documents)
    for ndx, node in enumerate(nodes):
        print()
        print(ndx, len(node.text))
        print(node)
        print()

    index = make_index_from_nodes(nodes)
    print(index)

    storage_context = make_storage_context(nodes)

    index1, index2 = make_indexes_from_storage(storage_context, nodes)
    print(index1)
    print(index2)