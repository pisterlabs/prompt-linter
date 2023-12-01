import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


st.set_page_config(page_title="Chat with docs")

#streamlit run xxx.py


#PS:
#Dont' t know why there is often a problem of "KeyError: 'langchain'". But not every time. Maybe because we cannot run two streamlit together

#Test Cases：
#
#1. A txt file with short sentences:
### How do you think about my name?
### As an AI language model, I don't have personal opinions or feelings. However, I think your name "Andrew" is a common and classic name.

#2. My personal CV
### Hi, what do you think about my IEP project?
###I'm sorry, but I don't have the ability to form opinions as I am an AI language model. However, based on the information provided, it seems like you gained valuable experience using PicoScope and LTSpice, as well as investigating the characteristics of various electronic components. This will likely be helpful in your future studies and career in engineering.

# Get API Key
user_api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",
    placeholder="Paste your openAI API key, sk-",
    type="password")
os.environ["OPENAI_API_KEY"] = user_api_key


def get_text(doc):
    """
    Extracts the text from the provided documents.

    Args:
        doc (): uploaded file.

    Returns:
        str: text extracted from the Doc.
    """
    text = ""
    print(doc)
    doc_type = doc.type

    # PDF reader to load doc
    if doc_type == "application/pdf":
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Todo: add txt reader
    elif doc_type == "text/plain":
        text= doc.read().decode() 

    else:
        st.header("Non supported doc")
        text = ""

    return text


def get_text_chunks(text):
    """
    Splits the text into smaller chunks for efficient processing.

    Args:
        text (str): Input text.

    Returns:
        list: List of text chunks.
    """

    # Todo: apply CharacterTextSplitter() overhere
    #text_splitter = None
    #chunks = text_splitter.split_text(text)
    text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 0)
    chunks=text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Generates a vector store from the text chunks.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        object: Vector store object.
    """

    # Todo: apply OpenAIEmbedding() and FAISS.from_texts(texts, embedding) Here
    #text_embeddings = embeddings.embed_documents(text_chunks)
    #text_embedding_pairs = list(zip(text_chunks, text_embeddings))
    #faiss = FAISS.from_embeddings(text_embedding_pairs, embeddings) "Wrong"

    embeddings = OpenAIEmbeddings()
    faiss = FAISS.from_texts(text_chunks, embeddings)
    vectorstore=faiss
    #embeddings = None
    #vectorstore = None

    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Creates a conversational retrieval chain for handling user queries.

    Args:
        vectorstore (object): Vector store object.

    Returns:
        object: Conversational retrieval chain object.
    """

    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo") #model name!!
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
     # Todo: apply ConversationalRetrievalChain.from_llm(llm, retriever, memory)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)
    return conversation_chain


def handle_userinput(user_question):
    """
    Handles the user's input question and generates a response.

    Args:
        user_question (str): User's question.
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(message.content)
        else:
            st.write(message.content)


def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with docs")
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your document")
        input_doc = st.file_uploader(
            "Upload your file here and click on 'Process'", accept_multiple_files=False)
        # Todo: 
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_text(input_doc)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()