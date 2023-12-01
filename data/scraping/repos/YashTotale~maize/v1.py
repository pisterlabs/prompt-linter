import os
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ListIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    node_parser,
    LLMPredictor,
    ServiceContext,
    ResponseSynthesizer,
    EmptyIndex,
    KnowledgeGraphIndex,
    TreeIndex,
)


from flask import request, render_template, send_file
from langchain.llms import OpenAI

from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.graph_stores import SimpleGraphStore

from pyvis.network import Network

import networkx as nx
from io import BytesIO
import matplotlib.pyplot as plt

from flask import Flask
import time


PORT = 5003
app = Flask(__name__)

# NOTE: for local testing only, do NOT deploy with your key hardcoded
# kailash's: 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
# sasvath's api key: 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
# sasvath's second api key: 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
os.environ["OPENAI_API_KEY"] = "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"

granary_dir = "./granary"  # directory to store files
storage_dir = "./storage"
html_file = "./example.html"

storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
index: VectorStoreIndex = None  # type: ignore
kindex: KnowledgeGraphIndex = None
tindex: TreeIndex = None


@app.route("/api/sample-graph", methods=["GET"])
def random():
    return render_template(html_file)


@app.route("/api/knowledge", methods=["GET"])
def init_knowledge_index():
    global kindex
    graph_store = SimpleGraphStore()

    graph_storage_context = StorageContext.from_defaults(graph_store=graph_store)
    text_chunks = [
        granary_dir + "/berkeley.txt",
        granary_dir + "/california.txt",
        granary_dir + "/san-francisco.txt",
    ]

    documents = []
    count = 1
    for filename in text_chunks:
        with open(filename, "r") as file:
            documents.append(
                Document(
                    file.read(),
                    doc_id=f"doc_id_{count}",
                )
            )
        count += 1

    # documents = SimpleDirectoryReader(granary_dir, filename_as_id=True).load_data()
    llm_predictor = LLMPredictor(llm=OpenAI(model_name="gpt-4"))
    graph_service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    parser = node_parser.SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print("created document nodes")
    for node in nodes:
        print(node.ref_doc_id)
    print(len(nodes))

    time_to_create_index = time.time()
    kindex = KnowledgeGraphIndex(
        nodes=nodes,
        max_triplets_per_chunk=5,
        storage_context=graph_storage_context,
        service_context=graph_service_context,
    )
    time_to_create_index = time.time() - time_to_create_index

    graph_storage_context.persist(persist_dir=storage_dir)
    time_to_create_graph = time.time()

    graph = kindex.get_networkx_graph(text_chunks)
    time_to_create_graph = time.time() - time_to_create_graph
    net = Network(notebook=False, cdn_resources="in_line", directed=True)
    net.from_nx(graph)
    print(
        f"TIMING SUMMARY\nIndex Creation: {time_to_create_index}s\nGraph building: {time_to_create_graph}s\n"
    )
    return net.generate_html(), 200


@app.route("/api/tree", methods=["GET"])
def init_tree_index():
    global tindex
    # tree_store = SimpleGraphStore()
    # tree_storage_context = StorageContext.from_defaults(graph_store=graph_store)
    documents = SimpleDirectoryReader(granary_dir).load_data()
    llm_predictor = LLMPredictor(llm=OpenAI(model_name="gpt-4"))
    tree_service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    parser = node_parser.SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(len(nodes))

    time_to_create_index = time.time()
    tindex = TreeIndex(
        nodes=nodes,
        num_children=3,
        # storage_context=graph_storage_context,
        service_context=tree_service_context,
        build_tree=True,
    )
    time_to_create_index = time.time() - time_to_create_index
    print(tindex)

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


@app.before_request
def before_request():
    global index, storage_context

    if os.path.exists(storage_dir):
        index = load_index_from_storage(storage_context)  # type: ignore
    else:
        documents = SimpleDirectoryReader(granary_dir).load_data()
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        storage_context.persist()


# filename_fn = lambda filename: {"file_name": filename}
# documents = SimpleDirectoryReader("./granary", filename_as_id=True).load_data()
# assert len(documents) == 2

# # index = VectorStoreIndex.from_documents(documents)
# parser = node_parser.SimpleNodeParser()
# nodes = parser.get_nodes_from_documents(documents)

# llm_predictor = LLMPredictor(llm=OpenAI(model_name="gpt-4"))
# service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# index = VectorStoreIndex(nodes, service_context=service_context)
# storage_context.docstore.add_documents(nodes)


@app.route("/")
def home():
    return "Welcome to the Maize API. Use /api/ to call API routes."


@app.route("/api/")
def home2():
    return "nice"


@app.route("/api/granary", methods=["GET"])
def granary():
    global index, storage_context

    kernels = [
        os.path.join(granary_dir, f)
        for f in os.listdir(granary_dir)
        if os.path.isfile(os.path.join(granary_dir, f))
    ]
    kernel_info = []
    for kernel in kernels:
        saved_file = open(kernel, "r")
        file_content = saved_file.read()
        pass

    return {
        "success": True,
        "message": "Success fetching granary!",
        "payload": {"kernel_info": kernel_info},
    }, 200


@app.route("/api/createKernel", methods=["POST"])
def createKernel():
    global index, storage_context

    file = request.files.get("file")
    if file is None:
        return "File required", 400

    file_path = os.path.join(granary_dir, file.filename)  # type: ignore
    file.save(file_path)
    saved_file = open(file_path, "r")
    file_content = saved_file.read()

    doc = Document(file_content)
    index.insert(doc)

    index.storage_context.persist(persist_dir=storage_dir)

    return {
        "success": True,
        "message": "Success creating kernel!",
        "payload": {
            "kernel_id": doc.doc_id,
        },
    }, 200


@app.route("/api/query", methods=["GET"])
def query_index():
    global index

    if index is None:
        return "No index found. Please upload documents to query."

    query_text = request.args.get("text", None)
    if query_text is None:
        return "No text found, please include a ?text=SIUUU parameter in the URL", 400
    # query_engine = index.as_query_engine()

    # response = query_engine.query(query_text)

    print(query_text)

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=4,
    )

    # configure response synthesizer
    response_synthesizer = ResponseSynthesizer.from_args(
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
    )

    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    # query
    response = query_engine.query(query_text)

    print("SOURCES: " + response.get_formatted_sources())

    return {
        "success": True,
        "message": "Success querying!",
        "payload": {
            "response": response.response,  # type: ignore
            "response_nodes": response.source_nodes,
        },
    }, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
