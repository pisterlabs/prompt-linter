import os
import pinecone
import tempfile
from typing import Any, Dict, Optional, Sequence, Union
from flask import Flask, request
from llama_index import (
    VectorStoreIndex,
    Document,
    StorageContext,
    ResponseSynthesizer,
    KnowledgeGraphIndex,
    TreeIndex,
    LLMPredictor,
    load_index_from_storage,
    load_graph_from_storage,
    node_parser,
    ServiceContext,
    EmptyIndex,
    SimpleDirectoryReader,
)

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.graph_stores import SimpleGraphStore
from llama_index.indices.base import BaseIndex
from langchain.llms import OpenAI
from flask import request, render_template, send_file

from pyvis.network import Network

import time

from llama_index.vector_stores import PineconeVectorStore
import json

PORT = 5000
app = Flask(__name__)

os.environ["STORAGE_DIR"] = "./storage"
os.environ["GRANARY_DIR"] = "./granary"
os.environ["TEMP_DIR"] = "./temp"
os.environ["FILES_DB"] = "./db.json"
os.environ["VECTOR_DIM"] = "1536"
os.environ["PINECONE_API_KEY"] = "b5c18ed3-b2fe-407b-a737-83e14e23fc63"
os.environ["PINECONE_ENVIRONMENT"] = "asia-southeast1-gcp-free"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SEARCH_THRESHOLD"] = "0.76"
KNOWLEDGE_STORAGE_DIR = "./kstorage"
TREE_STORAGE_DIR = "./tstorage"


parser = node_parser.SimpleNodeParser()


def init_pinecone():
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )
    pinecone_index = pinecone.Index("maize")
    pinecone_vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return pinecone_index, pinecone_vector_store


pinecone_index, pinecone_vector_store = init_pinecone()


def init_index():
    global pinecone_vector_store

    storage_context = StorageContext.from_defaults(
        vector_store=pinecone_vector_store,
    )
    index = VectorStoreIndex.from_documents([], storage_context=storage_context)
    return storage_context, index


storage_context, index = init_index()


@app.route("/", methods=["GET"])
def home():
    return "Home", 200


# Given a doc_id, returns the text of the document
def get_granary_text(doc_id):
    doc_path = os.path.join(os.environ["GRANARY_DIR"], doc_id + ".txt")
    granary_reader = open(doc_path, "r")
    file_content = granary_reader.read()
    granary_reader.close()

    return file_content


# Returns all files in db.json with their full text appended
def get_all_files():
    # Get files database
    files_db_reader = open(os.environ["FILES_DB"], "r")
    files_db: Dict[str, Dict[str, str]] = json.load(files_db_reader)
    files_db_reader.close()

    # Create version of files that has full text appended as an attribute for each file
    for doc_id in files_db:
        #' Read full text and assign to 'text' attribute'
        files_db[doc_id]["text"] = get_granary_text(doc_id)

    return files_db


# Given a doc_id, get the corresponding filename using the db.json file
def query_files_db(doc_id):
    files_db_reader = open(os.environ["FILES_DB"], "r")
    files_db = json.load(files_db_reader)
    files_db_reader.close()

    return files_db[doc_id]["filename"]


def generate_doc_from_file(doc_id: str, file: Dict[str, str]):
    filename = file["filename"]
    filecontent = file["text"]
    return Document(
        filecontent,
        doc_id=f"{filename} ({doc_id})",
    )


def generate_docs_from_files(files: Dict[str, Dict[str, str]]):
    documents = []
    for doc_id in files.keys():
        documents.append(generate_doc_from_file(doc_id, files[doc_id]))
    return documents


# @app.route("/<int:doc_id>", methods=["DELETE"])
# def delete_document(doc_id):
#     # pinecone.delete
#     pass


@app.route("/api/relationMap", methods=["GET"])
def create_relation_map():
    global kindex
    graph_store = SimpleGraphStore()

    text_chunks = get_all_files()

    documents = generate_docs_from_files(text_chunks)

    # documents = SimpleDirectoryReader(granary_dir, filename_as_id=True).load_data()
    llm_predictor = LLMPredictor(llm=OpenAI(model_name="gpt-4"))
    graph_service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    nodes = parser.get_nodes_from_documents(documents)
    print("created document nodes")
    for node in nodes:
        print(node.ref_doc_id)
    print(len(nodes))

    time_to_create_index = time.time()

    kindex = None
    graph_storage_context = None
    graph = None
    if os.path.exists(KNOWLEDGE_STORAGE_DIR):
        graph_store = SimpleGraphStore.from_persist_dir(KNOWLEDGE_STORAGE_DIR)

        graph_storage_context = StorageContext.from_defaults(
            persist_dir=KNOWLEDGE_STORAGE_DIR, graph_store=graph_store
        )
        kindex = load_index_from_storage(graph_storage_context)  # type: ignore
        # graph = load_graph_from_storage(graph_storage_context, root_id=kindex.index_id)
        print(kindex.index_struct.table.keys())
        print(kindex.index_struct.node_mapping.keys())
        print(type(kindex))
    else:
        graph_storage_context = StorageContext.from_defaults(graph_store=graph_store)
        kindex = KnowledgeGraphIndex(
            nodes=nodes,
            max_triplets_per_chunk=5,
            storage_context=graph_storage_context,
            service_context=graph_service_context,
        )
        graph_storage_context.persist(persist_dir=KNOWLEDGE_STORAGE_DIR)

    graph = kindex.get_networkx_graph(text_chunks)  # type: ignore

    time_to_create_index = time.time() - time_to_create_index

    # time_to_create_graph = time.time()

    # time_to_create_graph = time.time() - time_to_create_graph
    net = Network(notebook=False, cdn_resources="in_line", directed=True)
    net.from_nx(graph)
    net.show_buttons(filter_=["physics"])
    net.save_graph("example.html")  # Backup
    print(
        f"TIMING SUMMARY\nIndex Creation: {time_to_create_index}s\nGraph building: N/A s\n"
    )
    return net.generate_html(), 200


@app.route("/api/tree", methods=["GET"])
def init_tree_index():
    global tindex
    # tree_store = SimpleGraphStore()
    # tree_storage_context = StorageContext.from_defaults(graph_store=graph_store)
    text_chunks = get_all_files()

    documents = generate_docs_from_files(text_chunks)

    llm_predictor = LLMPredictor(llm=OpenAI(model_name="gpt-3.5-turbo"))
    tree_service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    parser = node_parser.SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(len(nodes))

    time_to_create_index = time.time()
    tindex = TreeIndex.from_documents(documents, service_context=tree_service_context)
    print("created initial tree index")

    # tindex = TreeIndex(
    #     documents[0],
    #     num_children=3,
    #     service_context=tree_service_context,
    #     build_tree=True,
    # )
    # for d in documents:
    #     if d == documents[0]:
    #         continue
    #     tindex.insert(d)
    #     print(f"Inserted document into tree index")

    # tindex = TreeIndex(
    #     nodes=nodes,
    #     num_children=3,
    #     # storage_context=graph_storage_context,
    #     service_context=tree_service_context,
    #     build_tree=True,
    # )
    time_to_create_index = time.time() - time_to_create_index
    # print(tindex)

    print(f"TIMING SUMMARY\nIndex Creation: {time_to_create_index}s\n")

    return "successful", 200

    # return {
    #     "success": True,
    #     "message": "Success querying!",
    #     "payload": {
    #         "response": response.response,  # type: ignore
    #         "response_nodes": response.source_nodes,
    #     },
    # }, 200


@app.route("/api/granary", methods=["GET"])
def query_index():
    query_text = request.args.get("query", None)

    # If no query, then get all documents.
    if query_text is None or query_text == "":
        files_db = get_all_files()
        return {
            "success": True,
            "message": "Success querying!",
            "payload": {"relevantDocs": files_db},
        }, 200

    else:
        # Define empty response
        relevant_docs = {}

        # GET THE QUERY RESPONSE

        # configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=4,
        )

        # configure response synthesizer
        response_synthesizer = ResponseSynthesizer.from_args(
            node_postprocessors=[
                SimilarityPostprocessor(
                    similarity_cutoff=float(os.environ["SEARCH_THRESHOLD"])
                )
            ]
        )

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        # If query text given, provide the query response
        query_res = query_engine.query(query_text)

        # store only the text of the query response
        text_res = query_res.response  # type: ignore

        # Go through each document that the response used as a source
        nodes = query_res.source_nodes

        for obj in nodes:
            node = obj.node
            # If node is relevant enough...
            if obj.score >= float(os.environ["SEARCH_THRESHOLD"]):  # type: ignore
                # Get node info
                node_info = node.node_info or {"start": 0, "end": 0}

                if node.ref_doc_id in relevant_docs:
                    # Add node info to existing doc object
                    relevant_docs[node.ref_doc_id]["nodes"].append(node_info)
                else:
                    # Read text of the node
                    text = get_granary_text(node.ref_doc_id)

                    # Get filename corresponding to doc_id
                    filename = query_files_db(node.ref_doc_id)

                    # Add object to relevantDocs object with its filename and full text
                    relevant_docs[node.ref_doc_id] = {
                        "filename": filename,
                        "text": text,
                        "nodes": [node_info],
                    }
                # Read text of the node
                text = get_granary_text(node.ref_doc_id)

                # Get filename corresponding to doc_id
                filename = query_files_db(node.ref_doc_id)

                # Add object to relevantDocs object with its filename and full text
                relevant_docs[node.ref_doc_id] = {
                    "filename": filename,
                    "text": text,
                    "nodes": [node.node_info],
                }

        return {
            "success": True,
            "message": "Success querying!",
            "payload": {
                "relevantDocs": relevant_docs,  # type: ignore
                "textResponse": text_res,
            },
        }, 200


@app.route("/api/createKernel", methods=["POST"])
def createKernel():
    global index, storage_context, kindex, parser

    file = request.files.get("file")
    if file is None:
        return {"success": False, "message": "File required"}, 400

    # Temporarily save the file in the temp directory to read its contents
    temp_file_path = os.path.join(os.environ["TEMP_DIR"], file.filename)  # type: ignore
    file.save(temp_file_path)
    file.close()
    
    temp_reader = open(temp_file_path, "r")
    file_content = temp_reader.read()
    temp_reader.close()

    # Create document and insert into index
    doc = Document(file_content)
    index.insert(doc)
    doc_id = doc.get_doc_id()

    # Remove the temp file from the temp directory
    os.remove(temp_file_path)

    # open existing map of files
    files_db_reader = open(os.environ["FILES_DB"], "r")
    files_db = json.load(files_db_reader)
    files_db_reader.close()

    # Connect generated doc_id to user_specified filepath
    files_db[doc_id] = {"filename": file.filename}

    # Save the file in the actual directory
    file_path = os.path.join(os.environ["GRANARY_DIR"], doc_id + ".txt")  # type: ignore
    file_writer = open(file_path, "w")
    file_writer.write(file_content)
    file_writer.close()

    # Write to the map of files
    files_db_writer = open(os.environ["FILES_DB"], "w")
    json.dump(files_db, files_db_writer)
    files_db_writer.close()

    # Add to knowledge index
    doc_with_metadata = generate_doc_from_file(
        doc_id,
        {
            "filename": file.filename,  # type: ignore
            "text": file_content,
        },
    )

    llm_predictor = LLMPredictor(llm=OpenAI(model_name="gpt-4"))
    graph_service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    kindex = None
    graph_storage_context = None
    if os.path.exists(KNOWLEDGE_STORAGE_DIR):
        graph_store = SimpleGraphStore.from_persist_dir(KNOWLEDGE_STORAGE_DIR)
        graph_storage_context = StorageContext.from_defaults(
            persist_dir=KNOWLEDGE_STORAGE_DIR, graph_store=graph_store
        )
        kindex = load_index_from_storage(graph_storage_context)  # type: ignore
        # print(kindex.index_struct.table.keys())
        # print(kindex.index_struct.node_mapping.keys())
        # print(type(kindex))
        print(len(kindex.index_struct.node_mapping.keys()))
        kindex.insert(doc_with_metadata)
        print(len(kindex.index_struct.node_mapping.keys()))
    else:
        graph_store = SimpleGraphStore()
        graph_storage_context = StorageContext.from_defaults(graph_store=graph_store)
        kindex = KnowledgeGraphIndex.from_documents(
            [doc_with_metadata],
            max_triplets_per_chunk=5,
            storage_context=graph_storage_context,
            service_context=graph_service_context,
        )

    graph_storage_context.persist(persist_dir=KNOWLEDGE_STORAGE_DIR)

    return {
        "success": True,
        "message": "Success creating kernel!",
        "payload": {},
    }, 200


if __name__ == "__main__":
    app.run(port=PORT)
