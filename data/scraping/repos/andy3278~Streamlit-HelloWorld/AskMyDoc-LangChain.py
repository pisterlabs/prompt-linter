import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from pypdf import PdfReader

# Build an Ask the Doc app

st.set_page_config(page_title="LangChain Ask the Doc", page_icon="🦜🔗", layout="wide")
st.title("LangChain Ask the Doc")

uploaded_file = st.file_uploader("Upload a pdf file", type="pdf")

st.write("This app needs OpenAI API key to work. Please enter your API key below \n\n the key should start with 'sk-'")
# api_key should not be visible to the user, it should be like *****
api_key = st.text_input("API Key", type="password")
openai.api_key = api_key

# disable the question input if no file is uploaded
question = st.text_input("Enter your question here: ", placeholder="What is the summary of the document?", disabled = not uploaded_file)


def load_pdf_data(uploaded_file):
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text


def generate_response(uploaded_file, api_key, question):
    if uploaded_file is not None:
        st.write("File uploaded")
        doc = load_pdf_data(uploaded_file)
        # split the text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(doc)

        # create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        # create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # create retriever interface
        retriever = db.as_retriever()
        # create QA chain
        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=api_key), chain_type="stuff", retriever=retriever)
    return qa.run(question)
    
result = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# iterate through the messages in the session state
# and display them in the chat message container
# identify user and assistant messages by their role
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if question:
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # add message to history
    st.session_state.messages.append({"role": "user", "content": question})

    response = generate_response(uploaded_file, api_key, question)
    with st.chat_message("assistant"):
        st.markdown(response)

    # add the echo message to chat history
    st.session_state.messages.append({"role":"assistant","content":response})
