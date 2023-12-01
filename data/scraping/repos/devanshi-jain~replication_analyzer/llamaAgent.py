from llama_index import (
    VectorStoreIndex, 
    SimpleKeywordTableIndex, 
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from llama_index.indices.list import GPTListIndex

from langchain.llms.openai import OpenAIChat
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
import openai
import os

from langchain.document_loaders import PyPDFLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.chains import ConversationalRetrievalChain # for chatting with the pdf
from langchain.llms import OpenAI # the LLM model we'll use (CHatGPT)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) #reads local .env file

# Get value of OPENAI_API_KEY environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for the OpenAI API client

openai.api_key = openai_api_key

def llamaAgent(pdfAPath, pdfBPath):

    loader = PyPDFLoader(pdfAPath)
    pagesA = loader.load_and_split()
    loader = PyPDFLoader(pdfBPath)
    pagesB = loader.load_and_split()

    # llm_predictor_chatgpt = LLMPredictor(llm=OpenAIChat(temperature=0, model_name="gpt-3.5-turbo-16k"))
    # service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt)

    directCompareIndices = []
    for a in range(len(pagesA)):
        for b in range(len(pagesB)):
            print(pagesA[b])
            vectordb = Chroma.from_documents([pagesA[a],pagesB[b]], embedding=embeddings, persist_directory=".")
            pdf_qa = C.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"), vectordb, return_source_documents=True)
            query = "Quick 2 sentence summary."
            result = pdf_qa({"question": query, "chat_history": ""})
            print("Answer:")
            print(result["answer"])
            query = "Is the information presented on a single topic? You must answer with an integer between 0 to 10."
            result = pdf_qa({"question": query, "chat_history": ""})
            print("Answer:")
            print(result["answer"])
            try:
                if int(result["answer"]) >= 9:
                    directCompareIndices.append((a, b))
            except:
                continue

    # indices = {}
    # descs = {}
    # for paper in ["A","B"]:
    #     indices[paper] = VectorStoreIndex.from_documents(pdfs[paper], service_context=service_context)
    #     descs[paper] = f"Paper {paper}"

    # graph = ComposableGraph.from_indices(
    #     SimpleKeywordTableIndex,
    #     [index for _, index in indices.items()], 
    #     [desc for _, desc in descs.items()],
    #     max_keywords_per_chunk=50
    # )
    # decompose_transform = DecomposeQueryTransform(
    #     llm_predictor_chatgpt, verbose=True
    # )

    # index = GPTListIndex.from_documents(pdfs, service_context=service_context)
    # query_engine = index.as_query_engine()

    # custom_query_engines = {}
    # for index in indices.values():
    #     query_engine = index.as_query_engine(service_context=service_context)
    #     transform_extra_info = {'index_summary': index.index_struct.}
    #     tranformed_query_engine = TransformQueryEngine(query_engine, decompose_transform, transform_extra_info=transform_extra_info)
    #     custom_query_engines[index.index_id] = tranformed_query_engine

    # custom_query_engines[graph.root_index.index_id] = graph.root_index.as_query_engine(
    #     retriever_mode='default', 
    #     response_mode='compact', 
    #     service_context=service_context
    # )

    # query_engine_decompose = graph.as_query_engine(
    #     custom_query_engines=custom_query_engines,
    # )

<<<<<<< HEAD
    prompt = f"""
        How correlated is Paper A with Paper B? You must respond with an integer between 0 to 10.
        """
    response = query_engine_decompose.query(prompt)
    print(response)
=======

    # prompt = f"""
    #     How correlated is Paper A with Paper B? You must respond with an integer between 0 to 10.
    #     """
    # response = query_engine_decompose.query(prompt)
    # print(response)
>>>>>>> ab64c7f2672ccd0d841f1048a314e59b6e45e518

    # prompt = f"""
    #     How do Paper B and Paper A relate?
    #     """
    # response = query_engine_decompose.query(prompt)
    # print(response)

    return 0


# Data: Does the second paper get the same results using the first paper's data?
# Method: Does the second paper use the same methods as the first?
# Outcome: Are the results of both papers similar?
# Verification: Can others confirm the second paper's results?
# Relevance: Is the second paper's work closely related to the first's?

if __name__ == "__main__":
    llamaAgent("./pdfA.pdf", "./pdfB.pdf")