import os
# needed for llm in from_chain_type()
import openai
import textwrap
# needed for editing questions on command line. Also allows to arrow up for previous questions
import readline

from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.chains import RetrievalQA


# InstructorEmbedding
from langchain.embeddings import HuggingFaceInstructEmbeddings


# Functions for output formating

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\nSources:')
    for source in llm_response["source_documents"]:
        print(f"{source.metadata['source']}: Page {source.metadata['page']}")




# load variables from .env
load_dotenv()


instructor_embeddings = HuggingFaceInstructEmbeddings(
                            model_name=os.getenv('EMBEDDINGS_MODEL'), 
                            model_kwargs={"device": "cuda" }
)

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.getenv('PGVECTOR_DRIVER'),
    host=os.getenv('PGVECTOR_HOST'),
    port=os.getenv('PGVECTOR_PORT'),
    database=os.getenv('PGVECTOR_DATABASE'),
    user=os.getenv('PGVECTOR_USER'),
    password=os.getenv('PGVECTOR_PASSWORD')
)


# Pull in details to query pgvector database

# Possible distance functions:
# EUCLIDEAN = EmbeddingStore.embedding.l2_distance
# COSINE = EmbeddingStore.embedding.cosine_distance
# MAX_INNER_PRODUCT = EmbeddingStore.embedding.max_inner_product

store = PGVector(
    connection_string=CONNECTION_STRING, 
    embedding_function=instructor_embeddings, 
    collection_name=os.getenv('COLLECTION_NAME'),
    distance_strategy=DistanceStrategy.COSINE
)

retriever = store.as_retriever()

#create the chain to answer questions 
qa_chain_instrucEmbed = RetrievalQA.from_chain_type(llm=OpenAI(temperature=1, ), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)



# main program loop
try:
    while True:
        query = input("Enter your query (or 'ctrl-c' to exit): ")
        llm_response = qa_chain_instrucEmbed(query)
        print(f"Query: {query}")
        print("Answer")
        process_llm_response(llm_response)
except KeyboardInterrupt:
    print("\nExiting the program.")

