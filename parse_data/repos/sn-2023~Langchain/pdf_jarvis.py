import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
# for creating embeddings and inserting them into a table in SingleStore
import sqlalchemy as db
import os
from sqlalchemy import text as sql_text
from collections import deque

#Initialize OpenAIEmbeddings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)#TODO: replace with your API key

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

#this method accepts a list of text chunks and returns a vectorstore
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

#function that takes a list of text chunks, creates embeddings and inserts them into a table in SingleStore
def create_embeddings_and_insert(text_chunks):
    ss_password = os.environ.get("SINGLESTORE_PASSWORD")
    ss_host = os.environ.get("SINGLESTORE_HOST")
    ss_user = os.environ.get("SINGLESTORE_USER")
    ss_database = os.environ.get("SINGLESTORE_DATABASE")
    ss_port = os.environ.get("SINGLESTORE_PORT")
    connection = db.create_engine(
        f"mysql+pymysql://{ss_user}:{ss_password}@{ss_host}:{ss_port}/{ss_database}")
    with connection.begin() as conn:
        # Iterate over the text chunks
        for i, text in enumerate(text_chunks):
            # Convert the text to embeddings
            embedding = embedder.embed_documents([text])[0]

            # Insert the text and its embedding into the database
            stmt = sql_text("""
                INSERT INTO multiple_pdf_example (
                    text,
                    embeddings
                )
                VALUES (
                    :text,
                    JSON_ARRAY_PACK_F32(:embeddings)
                )
            """)

            conn.execute(stmt, {"text": str(text), "embeddings": str(embedding)})


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# function to get similar text from the SingleStore embeddings table
def get_most_similar_text(query_text):
    # Convert the query text to embeddings
    query_embedding = embedder.embed_documents([query_text])[0]

    # Perform a similarity search against the embeddings
    stmt = sql_text("""
        SELECT
            text,
            DOT_PRODUCT_F32(JSON_ARRAY_PACK_F32(:embeddings), embeddings) AS similarity
        FROM multiple_pdf_example
        ORDER BY similarity DESC
        LIMIT 1
    """)

    ss_password = os.environ.get("SINGLESTORE_PASSWORD")
    ss_host = os.environ.get("SINGLESTORE_HOST")
    ss_user = os.environ.get("SINGLESTORE_USER")
    ss_database = os.environ.get("SINGLESTORE_DATABASE")
    ss_port = os.environ.get("SINGLESTORE_PORT")
    connection = db.create_engine(
        f"mysql+pymysql://{ss_user}:{ss_password}@{ss_host}:{ss_port}/{ss_database}")
    with connection.begin() as conn:
        result = conn.execute(stmt, {"embeddings": str(query_embedding)}).fetchone()

    return result[0]

def truncate_table():

    # Perform a similarity search against the embeddings
    stmt = sql_text("""
        truncate table multiple_pdf_example
    """)

    ss_password = os.environ.get("SINGLESTORE_PASSWORD")
    ss_host = os.environ.get("SINGLESTORE_HOST")
    ss_user = os.environ.get("SINGLESTORE_USER")
    ss_database = os.environ.get("SINGLESTORE_DATABASE")
    ss_port = os.environ.get("SINGLESTORE_PORT")
    connection = db.create_engine(
        f"mysql+pymysql://{ss_user}:{ss_password}@{ss_host}:{ss_port}/{ss_database}")
    with connection.begin() as conn:
        result = conn.execute(stmt)
    return result



# new handle_userinput function that uses the SingleStore embeddings table
def handle_userinput(user_question):
    with st.spinner('Processing your question...'):
        most_similar_text = get_most_similar_text(user_question)
        
        # Pass the most similar text from the book as a part of the prompt to ChatGPT
        prompt = f"The user asked: {user_question}. The most similar text from the documents is: {most_similar_text}"
        
        #print prompt
        #st.write(prompt)

        response = st.session_state.conversation({'question': prompt})
        
        # Add the new messages at the beginning of the deque
        for message in reversed(response['chat_history']):
            st.session_state.chat_history.appendleft(message)

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)




def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = deque(maxlen=100)

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # pass the text chunks to create_embeddings_and_insert in order to create embeddings and insert them into a table in SingleStore                
                create_embeddings_and_insert(text_chunks)

                # Initialize the conversation chain here
                llm = ChatOpenAI()
                # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
                memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

                st.success('PDFs processed successfully!')
        st.subheader("Maintenance")
        if st.button("Truncate Existing Documents"):

            ## Code should be added to remove any documents listed in the upload area
            st.write("Truncating...")
            user_question = None ## Needs updated - trying to remove any questions in the box
            truncate_table()
            if "conversation" not in st.session_state:
                st.session_state.conversation = None ## unsure if this is needed - was getting odd error
            st.success('Table truncated successfully!')
            
    # Enable the user to ask a question only after the PDFs have been processed
    if st.session_state.conversation:
        if user_question:
            #st.write(user_question)
            handle_userinput(user_question)




#if __name__ == '__main__':
#    main()
