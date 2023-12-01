import os
import re
import json
import dotenv
import tiktoken
import logging
import streamlit as st
from html import unescape
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer("hkunlp/instructor-xl")
# print(model.max_seq_length)

# model.max_seq_length = 256

# Initialize the logger
logging.basicConfig(filename="app_log.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

faiss_index = None

# Prompt template for Hugging Face Instruct Embeddings
temp_prompt = """
Generate an answer to the user's question based on the given context. 
TOP_RESULTS: {context}
USER_QUESTION: {question}

Include as much information as possible in the answer.
If you didn't find the answer from context, just say you don't know.
Final Answer in English:
SOURCES:
"""

def selected_embed_case(embeddings):
    if embeddings == 0:
        return {"name": "huggingfaceembeddings", "embeddings": HuggingFaceEmbeddings()}
    elif embeddings == 1:
        return {"name": "openaiembeddings", "embeddings": OpenAIEmbeddings()}
    elif embeddings == 2:
        return {"name": "huggingfaceinstructembeddings", "embeddings": HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"})}
    else:
        return {"name": "huggingfaceembeddings", "embeddings": HuggingFaceEmbeddings()}

# Helper function to load FAISS index
def load_faiss_index(embedding_type):
    # Get the directory name based on the selected embedding
    embeddings_option = selected_embed_case(embedding_type)

    # index file path
    faiss_index_path = f"faiss_index_{embeddings_option['name'].lower()}"
    embeddings = embeddings_option['embeddings']
    logging.info(f"Searching FAISS index: {faiss_index_path}")

    global faiss_index
    if faiss_index is not None:
        logging.info(f"FAISS index already loaded {faiss_index}.")
        return faiss_index
    else:
        if os.path.exists(faiss_index_path):
            # Load the FAISS index from the specified directory
            faiss_index = FAISS.load_local(faiss_index_path, embeddings)
            return faiss_index
        else:
            logging.info(f"FAISS index directory for {embedding_type} not found.")
            return None

# Function to get answers using the selected embedding and FAISS index
def get_answers(question, embeddings, faiss_index):
    prompt_template = PromptTemplate(template=temp_prompt, input_variables=["context", "question"])

    docs = faiss_index.similarity_search(question)
    for doc in docs:
        logging.info(f"The similir document from FAISS index: {doc}\n\n")

    # fill the prompt template
    chain_type_kwargs = {"prompt": prompt_template}
    logging.info(f"\n\nThe prompt template made from FAISS index: {chain_type_kwargs}")
    
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), 
                                           chain_type="stuff", retriever=faiss_index.as_retriever(search_kwargs={'k': 6}), 
                                           chain_type_kwargs=chain_type_kwargs, return_source_documents=True)

    with get_openai_callback() as cb:
        #answeres = qa_chain.run(question)
        answeres = qa_chain({'query': question})
        logging.info(f"Answers          : {answeres}")
        logging.info(f"Total Tokens     : {cb.total_tokens}")
        logging.info(f"Prompt Tokens    : {cb.prompt_tokens}")
        logging.info(f"Completion Tokens: {cb.completion_tokens}")
        logging.info(f"Total Cost (USD) : ${cb.total_cost}")

    return answeres

# Function to compute a hash for a given code
def get_code_hash(code):
    code_hasher = _CodeHasher()
    code_hasher.update(code.encode())
    return code_hasher.hexdigest()

# Check if the app is in 'rerun' mode (user clicked the button)
def is_rerun():
    session_state = _get_session_state()
    current_code_hash = get_code_hash(st.script_runner.code)
    last_code_hash = session_state.last_code_hash

    if last_code_hash and last_code_hash == current_code_hash:
        return True

    session_state.last_code_hash = current_code_hash
    return False

# Get the session state
def _get_session_state():
    if not hasattr(st, '_custom_session_state'):
        st._custom_session_state = {}
    return st._custom_session_state

def parse_response(response):
    # Extracting the answer from the response
    answer = response["result"]
    logging.info(f"Response answer: {answer}")

    # Extracting the URLs from the source_documents
    source_documents = response["source_documents"]
    urls = []
    urls_set = set()
    for doc in source_documents:
        metadata = doc.metadata
        slug = metadata['slug']
        title = metadata['title']
        url = f"[{title}](https://trip101.com/article/{slug})"
        logging.info(f"Response source document - metadata - in loop: {metadata}, title - in loop: {title}, slug - in loop: {slug}, url: {url}")
        if url not in urls_set:
            urls_set.add(url)

    urls = sorted(urls_set)

    # Creating the final result
    result = {
        "answer": answer,
        "sources": urls
    }

    return result

def main():
    load_dotenv()

    st.set_page_config(page_title="Trippy Bot: Answers your questions with the knowledge of https://trip101.com", page_icon="🤖")

    # Initialize different embeddings based on the user's selection
    embedding_options = { "Hugging Face": 0, "OpenAI": 1 }
    #embedding_options = { "Hugging Face": 0, "OpenAI": 1, "Hugging Face Instruct": 2 }
    
    # Load the existing FAISS index
    selected_embedding = st.sidebar.radio("Select Embedding", list(embedding_options.keys()))
    embeddings = embedding_options[selected_embedding]
    logging.info(f"Embeddings type: {type(embeddings).__name__.lower()}")

    faiss_index = load_faiss_index(embeddings)

    # Streamlit app setup
    logging.info("Trippy bot application is being loaded...!")
    st.title("Trippy Bot 🤖: Answers your questions with the knowledge of https://trip101.com")

    logging.info("Trippy bot application's settings is being loaded...!")
    st.sidebar.header("Settings")

    # Display the history of questions and answers
    st.subheader("Question History")
    session_state = _get_session_state()
    if 'history' in session_state:
        for answer in session_state['history']:
            logging.info(f"Answers : ---From--History--- : type: {type(answer)} ques: {answer['answer']['sources']}")
            st.write("Question:", answer["question"])
            st.write("Answer:\n")
            st.markdown(answer['answer']['answer'])

            st.write("Source:\n")
            st.markdown("\n".join([f"- {url}" for url in answer['answer']['sources']]))
            st.write("-" * 50)

    # Streamlit app main section
    question = st.text_input("Enter your question:")

    # Button to get answers
    if st.button("Get Answers"):
        if question:
            # Call the function to get answers and highlight usage cost messages
            response = get_answers(question, embeddings, faiss_index)
            if response:
                answer = parse_response(response)

                st.write("Answer:\n")
                st.markdown(answer['answer'])

                st.write("Source:\n")
                st.markdown("\n".join([f"- {url}" for url in answer['sources']]))

                # Store the question and answer in the history
                session_state = _get_session_state()
                if 'history' not in session_state:
                    session_state['history'] = []
                session_state['history'].append({"question": question, "answer": answer})
            else:
                st.write("Error:", "Something went wrong! Please try again later.")
    else:
        st.warning("Please enter a question.")

if __name__ == '__main__':
    main()

