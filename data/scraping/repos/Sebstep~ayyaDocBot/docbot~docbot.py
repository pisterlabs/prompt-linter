import os, argparse
from dotenv import load_dotenv
import openai
from llama_index import (
    StorageContext,
    ServiceContext,
    set_global_service_context,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.llms import OpenAI
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from storageLogistics import build_new_storage
import json
import time

# # variables

# define LLM
llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)

# setup
load_dotenv()
storage_folder = os.getenv("STORAGE_FOLDER")
output_folder = os.getenv("OUTPUT_FOLDER")

args = False
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--build", type=bool, default=False)
args = parser.parse_args()

storage_context = StorageContext.from_defaults(persist_dir=f"./{storage_folder}")

if args.build:
    print("Building new storage...", flush=True)
    build_new_storage()
else:
    # load vector store
    print("Loading existing storage...", flush=True)
    index = load_index_from_storage(storage_context)


# configure service context
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context)

# RAG-Flow
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=6,
)

response_synthesizer = get_response_synthesizer(response_mode="refine")

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# prepare to save the output
all_responses_list = []
counter = 0
timestamp = time.strftime("%Y%m%d%H%M%S")
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, f"{timestamp}_chat_output.txt")


# start the chat
while True:
    my_query = input("User: ")
    if my_query.lower() == "exit":
        break

    response = query_engine.query(my_query)

    print(f"Agent: {response.response}", flush=True)
    print(f"Sources: {response.get_formatted_sources()}", flush=True)

    this_sources_list = []
    for source in response.source_nodes:
        source_dict = {
            "id": source.node.node_id,
            "text": source.node.text,
            "score": source.score,
        }
        this_sources_list.append(source_dict)

    this_response_dict = {
        "query": my_query,
        "response": response.response,
        "sources": this_sources_list,
    }

    all_responses_list.append(this_response_dict)

    with open(output_file, "w") as f:
        json.dump(all_responses_list, f)

    counter += 1
