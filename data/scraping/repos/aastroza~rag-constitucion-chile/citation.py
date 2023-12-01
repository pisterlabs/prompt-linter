import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llama_index.query_engine import CitationQueryEngine
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
)
from dotenv import load_dotenv
import openai


load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

def create_query_engine(documents_path = "data/documents/", persist_dir = "./citation"):

    service_context = ServiceContext.from_defaults(
    llm_predictor=LLMPredictor(llm=ChatOpenAI(model_name='gpt-4', temperature=0))
    )

    documents = SimpleDirectoryReader(documents_path).load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=persist_dir)

    query_engine = CitationQueryEngine.from_args(
        index, 
        similarity_top_k=3,
        # here we can control how granular citation sources are, the default is 512
        citation_chunk_size=1024 
    )

    return query_engine

def get_response(query_engine, prompt):

    response = query_engine.query(prompt)

    sources = []
    for node in response.source_nodes:
            [source, capitulo, articulo] = node.node.get_text().split('\n', 3)[0:3]
            sources.append(f'[{source.replace("Source ", "" ).replace(":", "")}] {capitulo}, {articulo}\n')

    result = {
        "question": prompt,
        "sources": sources,
        "answer": response
    }

    return result

def get_final_response(query, response_vigente, response_propuesta):
    template = """
            You are a Constitutional Lawyer. You are asked to give a brief response about 
            the diferences of two constitutions about this topic: {query}.

            The first constitution is the current one, and the second one is a proposed one.
            Always refer to the first constitution as Constitución Actual and the second one as Constitución Propuesta.

            The first constituion says the following about the topic: {first_response}.
            The second constituion says the following about the topic: {second_response}.

            Please detail the differences between the two constitutions about this topic.
            Please be concise and respond in spanish.
            
           """
    prompt = PromptTemplate(template=template, input_variables=["query", "first_response", "second_response"])
    llm_chain = LLMChain(prompt=prompt, llm=ChatOpenAI(model_name='gpt-4', temperature=0))

    final_response = llm_chain.predict(query=query, first_response=response_vigente.response, second_response=response_propuesta.response)

    return final_response

query_engine_vigente = create_query_engine(documents_path = "data/documents", persist_dir = "./citation")

query_engine_propuesta = create_query_engine(documents_path = "data/documents_propuesta", persist_dir = "./citation_propuesta")