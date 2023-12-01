from config import openaiapi
import openai
import os
from langchain.document_loaders import TextLoader
from llama_index import Document
import uuid
import hashlib
import logging
import sys
from llama_index import (
    ListIndex,
    TreeIndex,
    LLMPredictor,
    ServiceContext,
)
from langchain.chat_models import ChatOpenAI
from llama_index.node_parser import SimpleNodeParser


def load_and_convert_documents(file_path):
    # Load the documents using langchain
    loader = TextLoader(file_path)
    langchain_docs = loader.load()
    # Convert langchain documents to llama-index documents
    llama_docs = []
    for doc in langchain_docs:
        text = doc.page_content
        doc_id = str(uuid.uuid4())
        doc_hash = hashlib.sha1(text.encode()).hexdigest()
        file_name = doc.metadata["source"].split("/")[-1]
        llama_doc = Document(
            text=text,
            doc_id=doc_id,
            embedding=None,
            doc_hash=doc_hash,
            extra_info={
                "file_name": f"{os.path.splitext(os.path.basename(file_path))[0]}"
            },
        )
        llama_docs.append(llama_doc)

    return llama_docs


def summariser(file_path):
    # openai.api_key = openaiapi
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    docs = load_and_convert_documents(file_path)
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(docs)
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openaiapi)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = ListIndex(nodes, service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(
        """These documents are key points extracted from a government file. Please read the document, and give an exhaustive overview of all 
         major topics that were mentioned over their respective time period (if applicable),  alongwith the action that was proposed or taken. Finally,
           give an executive summary, alongwith all the actions that may still be pending their completion according to the documents  """
    )
    return response


"""file_path = "keypoints/Draft 6th PMC Minutes - Dairy Value Chain.txt"
summary = summariser(file_path)
new_file_path = f"./summary/{os.path.splitext(os.path.basename(file_path))[0]}.txt"
with open(new_file_path, "w") as file:
    file.write(str(summary))
print(summary)"""
