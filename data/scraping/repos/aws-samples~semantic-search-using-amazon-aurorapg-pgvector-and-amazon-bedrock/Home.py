import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import S3FileLoader
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import Docx2txtLoader
import boto3
from botocore.exceptions import ClientError
import tempfile
import time
import hashlib
import json
import secrets

#replace the secret_name and region_name with AWS secret manager where your credentials are stored
def get_secret():
    sm_key_name = 'enter-your-secret-key-name' #eg: 'aurorapg-db-credentials'
    region_name = "us-west-2"
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=sm_key_name
        )
    except ClientError as e:
        print(e)
    secret = get_secret_value_response['SecretString']
    return secret

def generate_session_id():
    t = int(time.time() * 1000)
    r = secrets.randbelow(1000000) 
    return hashlib.md5(bytes(str(t) + str(r), 'utf-8'), usedforsecurity=False).hexdigest()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=512,
        chunk_overlap=103,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# replace the model_id and region_name if you are trying to call a different bedrock model
def get_vectorstore(text_chunks):
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",region_name="us-west-2")
    try:
        if text_chunks is None:
            return PGVector(
                connection_string=CONNECTION_STRING,
                embedding_function=embeddings,
            )
        return PGVector.from_texts(texts=text_chunks, embedding=embeddings, connection_string=CONNECTION_STRING)
    except Exception as e:
        print(e)
        print(text_chunks)


def get_conversation_chain(vectorstore):
    llm = Bedrock(model_id="anthropic.claude-instant-v1",region_name="us-west-2")
    memory = ConversationBufferMemory(memory_key="chat_history", return_source_documents=True, return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    bot_template = "BOT : {0}"
    user_template = "USER : {0}"
    try:
        response = st.session_state.conversation({'question': user_question})
        print("Response",response)
    except ValueError as e:
        st.write(e)
        st.write("Sorry, please ask again in a different way.")
        return
    st.session_state.chat_history = response['chat_history']
    st.write(user_template.replace("{0}", response['question']))
    st.write(bot_template.replace("{0}", response['answer']))
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{0}", message.content))
        else:
            st.write(bot_template.replace(
                "{0}", message.content))


def main():

    st.title("Semantic search application leveraging pgvector, Aurora PostgreSQL and Amazon Bedrock")
    source = ['PDFs','S3 (txt)','Youtube','CSV','PPT','Word']
    selected_source = st.selectbox("Select a source", source)
    if selected_source=='PDFs':
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
    elif selected_source=='S3 (txt)':
        s3_client = boto3.client('s3')
        # the objects are stored in aurora-genai bucket. Enter the appropriate bucket name
        response = s3_client.list_objects_v2(Bucket='s3-bucket-name', Prefix='documentEmbeddings/')
        document_keys = [obj['Key'].split('/')[1] for obj in response['Contents']][1:]
        user_input = st.selectbox("Select an S3 document and click on 'Process'", document_keys)
        if st.button("Process"):
            with st.spinner("Processing"):
                prefix="documentEmbeddings/"+user_input
                loader = S3FileLoader("s3-bucket-name", prefix)
                docs = loader.load()
                for i in docs:
                    text_chunks = get_text_chunks(i.page_content)
                    vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)  
    elif selected_source=='Youtube':
        user_input = st.text_input("Enter an youtube link and click on 'Process'")
        if st.button("Process"):
            with st.spinner("Processing"):
                loader = YoutubeLoader.from_youtube_url(user_input)
                transcript = loader.load()
                for i in transcript:
                    text_chunks = get_text_chunks(i.page_content)
                    vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
    elif selected_source=='PPT':
        ppt_docs = st.file_uploader("Upload your PPT here and click on 'Process'", type=['ppt', 'pptx'], accept_multiple_files=False)
        if st.button("Process"):
            with st.spinner("Processing"):
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(ppt_docs.getvalue())
                    tmp_file_path = tmp_file.name
                loader = UnstructuredPowerPointLoader(tmp_file_path)
                docs = loader.load()
                for i in docs:
                    text_chunks = get_text_chunks(i.page_content)
                    vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
    elif selected_source=='Word':
        word_docs = st.file_uploader("Upload your Word file here and click on 'Process'", type=['docx'], accept_multiple_files=False)
        if st.button("Process"):
            with st.spinner("Processing"):
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(word_docs.getvalue())
                    tmp_file_path = tmp_file.name
                loader = Docx2txtLoader(tmp_file_path)
                docs = loader.load()
                for i in docs:
                    text_chunks = get_text_chunks(i.page_content)
                    vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(get_vectorstore(None))
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

# enter the appropriate DB name
if __name__ == '__main__':  
    secret = json.loads(get_secret())
    CONNECTION_STRING = PGVector.connection_string_from_db_params(                                                  
        driver = "psycopg2",
        user = secret["username"],                                      
        password = secret["password"],                                  
        host = secret["host"],                                            
        port = 5432,                                          
        database = "genai"                                      
    )
    main()