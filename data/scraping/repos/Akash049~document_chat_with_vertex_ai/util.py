from langchain.embeddings import VertexAIEmbeddings
import streamlit as st
import pinecone

PINECONE_API_KEY = "<INSERT_YOUR_PINECONE_API_KEY>"
PINECONE_API_ENV = "<INSERT_YOUR_PINECONE_ENVIRONMENT>"
PINECONE_INDEX_NAME = "<INSERT_YOUR_PINECONE_INDEX>"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME)

v_embeddings = VertexAIEmbeddings()


def find_match(input_query):
    input_em = v_embeddings.embed_query(input_query)
    result = index.query(input_em, top_k=5, includeMetadata=True)
    result_context = result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']
    return result_context


def create_context_using_document(docs):
    context_string = ""
    for doc in docs:
        context_string += doc.page_content + "\n\n"
    return context_string


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string


