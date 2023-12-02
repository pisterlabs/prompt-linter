import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

OPEN_AI_KEY = # put in key here
PINECONE_API_KEY = # put in key here
PINECONE_API_ENV = # put in env here

# gets the pdf text
def get_pdf_text(raw_pdf):
    text = ""
    pdf_reader = PdfReader(raw_pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# converts the texts into chunks so it is easier to process via langchain
def get_text_chunks(some_text):
    text_splitter = CharacterTextSplitter(separator = "\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(some_text)
    return chunks

# create the OpenAI embeddings that are needed
def create_embeddings(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key = OPEN_AI_KEY)
    text_chunk_embeddings = embeddings.embed_documents([text for text in text_chunks])
    return text_chunk_embeddings

# instantiate the GPT 3.5 turbo model

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  
    environment=PINECONE_API_ENV, 
    index_name = "quadstest1"
)

# pinecone store embeddings
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)
    vectorstore = docsearch = Pinecone.from_texts(text_chunks, embeddings, index_name = "quadstest1")
    return vectorstore

# create chain that keeps track of questions & memory
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=OPEN_AI_KEY)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# have a conversation and handle the user imput
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title = "quads")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("quads")
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_doc = st.file_uploader(
            "Upload your PDF here and click on 'Process'")
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_doc)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()