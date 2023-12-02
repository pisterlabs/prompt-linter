import json
import os
import streamlit as st
import datetime

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text) 

def on_like():
    with open('data/qa.json', 'a') as f:
        qa_json = {
            'question': st.session_state.query,
            'answer': st.session_state.answer,
            'sources': st.session_state.sources,
            'email': st.session_state.email,
            'position': st.session_state.position,
            'feedback': 1,
            'timestamp': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        f.write(json.dumps(qa_json))
        f.write('\n')

    st.session_state.query = ''
    st.session_state.answer = ''
    st.session_state.sources = []

    st.success('Thank you for your feedback!')

def on_dislike():
    with open('data/qa.json', 'a') as f:
        qa_json = {
            'question': st.session_state.query,
            'answer': st.session_state.answer,
            'sources': st.session_state.sources,
            'email': st.session_state.email,
            'position': st.session_state.position,
            'feedback': -1,
            'timestamp': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        f.write(json.dumps(qa_json))
        f.write('\n')

    st.session_state.query = ''
    st.session_state.answer = ''
    st.session_state.sources = []

    st.success('Thank you for your feedback!')

def run_chat():
    with open('content/manual.txt', 'r') as f:
        manual = f.read()

    st.expander('User Manual', expanded=False).markdown(manual)

    st.markdown("### Question")

    query = st.text_input(label='Query', label_visibility='hidden', key='query')

    ask_button = st.button("Ask")

    chat_name = "no-dev"
    documents_path = "data/documents/nd/"

    with open('content/prompt.txt', 'r') as f:
        system_template = f.read()

    user_template = r""" 
    {question}
    """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]

    PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(messages)

    # vectorstore
    if not os.path.exists(f"data/db/{chat_name}"):
        documents = []
        for filename in os.listdir(documents_path): 
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(os.path.join(documents_path, filename))
                documents.extend(loader.load())

        # splitter
        # TODO: use better splitter
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        print("Creating new vectorstore")
        vectorstore = Chroma.from_documents(texts, OpenAIEmbeddings(deployment='embeddings', chunk_size=16), collection_name=chat_name, persist_directory=f"data/db/{chat_name}")
        vectorstore.persist()
    else:
        print("Loading existing vectorstore")
        vectorstore = Chroma(collection_name=chat_name, embedding_function=OpenAIEmbeddings(deployment='embeddings', chunk_size=16), persist_directory=f"data/db/{chat_name}")

    if query:
        st.markdown("### Answer")

        chat_box = st.empty() 
        stream_handler = StreamHandler(chat_box)

        qa = ConversationalRetrievalChain.from_llm(
            AzureChatOpenAI(deployment_name='llm', model_name="gpt-4", temperature=0, streaming=True, callbacks=[stream_handler]),
            vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={'prompt': PROMPT_TEMPLATE})

        response = qa({'question': query, 'chat_history': []})

        st.session_state.answer = response['answer']
        st.session_state.sources = [doc.metadata['source'].split('/')[-1].replace('.pdf', '') for doc in response['source_documents']]

        feedback = st.columns([0.86, 0.07, 0.07])
        like = feedback[1].button("👍", on_click=on_like)
        dislike = feedback[2].button("👎", on_click=on_dislike)

        st.markdown("### Sources")

        for doc in response['source_documents']:
            st.markdown(f"- {doc.metadata['source'].split('/')[-1].replace('.pdf', '')}")

        with open('data/qa.json', 'a') as f:
            qa_json = {
                'question': query,
                'answer': response['answer'],
                'sources': [doc.metadata['source'].split('/')[-1].replace('.pdf', '') for doc in response['source_documents']],
                'email': st.session_state.email,
                'position': st.session_state.position,
                'timestamp': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
            f.write(json.dumps(qa_json))
            f.write('\n')


    st.markdown("""---""")
    expander = st.expander("All source documents in the database", expanded=False)

    list_of_files = []
    for f in os.listdir(documents_path):
        if f.endswith('.pdf'):
            list_of_files.append(f.replace('.pdf', ''))

    list_of_files.sort()
    for file in list_of_files:
        expander.markdown(f"- {file}")
