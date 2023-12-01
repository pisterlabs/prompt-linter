import streamlit as st
import datetime
import os
from os import environ
import PyPDF2
# from langchain.agents import AgentType, initialize_agent
# from langchain.agents.agent_toolkits import ZapierToolkit
# from langchain.llms import OpenAI
# from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pysqlite3
from langchain.chat_models import ChatOpenAI
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import qdrant_client
from qdrant_client import QdrantClient,models
from qdrant_client.http.models import PointStruct
# from langchain.agents import initialize_agent
# from langchain.vectorstores import Qdrant
# from zap import schedule_interview


openai_api_key = st.secrets["OPENAI_API_KEY"]
# QDRANT_COLLECTION ="resume"


# client = QdrantClient(
#     url="https://fd3fb6ff-e014-4338-81ce-7d6e9db358b3.eu-central-1-0.aws.cloud.qdrant.io:6333", 
#     api_key=st.secrets["QDRANT_API_KEY"],
# )

# # Get a list of all existing collections
# collections = client.get_collections()

# # Check if the collection exists before attempting to clear its data
# if QDRANT_COLLECTION in collections:
#     # Delete the collection and all its data
#     client.delete_collection(collection_name="QDRANT_COLLECTION")
    
# collection_config = qdrant_client.http.models.VectorParams(
#         size=1536,
#         distance=qdrant_client.http.models.Distance.COSINE
#     )
# client.recreate_collection(
#    collection_name=QDRANT_COLLECTION,
#     vectors_config=collection_config)



def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def generate_response(doc_texts, openai_api_key, query_text):

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1,openai_api_key=openai_api_key)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(doc_texts)
    
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever(search_type="similarity")
    #Bot memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    custom_prompt_template = """You are a hiring manager's helpful assistant that reads multiple resumes of candidates and answers the hiring manager's questions related to the candidates,\
                        Do your best to answer the hiring manager's so that it helps them select candidates better\
                        Your goal is to aid the hiring in the candidate selection process.
                        If you don't know the answer, just say that you don't know, don't try to make up an answer.\
                        If you are asked to summarize a candidate'sresume, summarize it in 7 sentences, 3 sentences for their experience, 2 sentences projects, 1 sentence for their education and 1 sentence for their skills\
                        If you are asked to compare certain candidates just provide the separate summarization of those candidate's resumes.\
                        If you are asked for the candidate's email, provide them with the candidate's email along with the candidate's name\
                        If you asked to select a candidates based on certain skills or experience then go through the resumes and find the candidates with the relevant skills or experience and provide the hiring manager with a list of those candidates/

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    
    docs = db.similarity_search(query_text)
    #Create QA chain 
    qa =  RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=retriever,
                                       return_source_documents=False,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    response = qa({'query': query_text})
    return response["result"]
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "You are a AI assistant created to help hiring managers review resumes and shortlist candidates. You have been provided with resumes and job descriptions to review. When asked questions, use the provided documents to provide helpful and relevant information to assist the hiring manager. Be concise, polite and professional. Do not provide any additional commentary or opinions beyond answering the questions directly based on the provided documents."}]

# Page title
st.set_page_config(page_title='Gforce Resume Assistant', layout='wide')
st.title('Gforce Resume Assistant')

# File upload
uploaded_files = st.file_uploader('Please upload you resume(s)', type=['pdf'], accept_multiple_files=True)

# Query text
query_text = st.text_input('Enter your question:', placeholder='Select candidates based on experience and skills')

# Initialize chat placeholder as an empty list
if "chat_placeholder" not in st.session_state.keys():
    st.session_state.chat_placeholder = []

# Form input and query
if st.button('Submit', key='submit_button'):
    if openai_api_key.startswith('sk-'):
        if uploaded_files and query_text:
            documents = [read_pdf_text(file) for file in uploaded_files]
            with st.spinner('Chatbot is typing...'):
                response = generate_response(documents, openai_api_key, query_text)
                st.session_state.chat_placeholder.append({"role": "user", "content": query_text})
                st.session_state.chat_placeholder.append({"role": "assistant", "content": response})

            # Update chat display
            for message in st.session_state.chat_placeholder:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        else:
            st.warning("Please upload one or more PDF files and enter a question to start the conversation.")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.chat_placeholder = []
    uploaded_files.clear()
    query_text = ""
    st.empty()  # Clear the chat display

st.button('Clear Chat History', on_click=clear_chat_history)

# Create a sidebar with text input boxes and a button
st.sidebar.header("Schedule Interview")
st.sidebar.subheader("\nUnfortunately the API to schedule an interview is no longer supported.\nSorry for the inconvenience", "")
person_name = st.sidebar.text_input("Enter Person's Name", "")
person_email = st.sidebar.text_input("Enter Person's Email Address", "")
date = st.sidebar.date_input("Select Date for Interview")
time = st.sidebar.time_input("Select Time for Interview")
schedule_button = st.sidebar.button("Schedule Interview")

if schedule_button:
    if not person_name:
        st.sidebar.error("Please enter the person's name.")
    elif not person_email:
        st.sidebar.error("Please enter the person's email address.")
    elif not date:
        st.sidebar.error("Please select the date for the interview.")
    elif not time:
        st.sidebar.error("Please select the time for the interview.")
    else:
        # Call the schedule_interview function from the zap.py file
        success = schedule_interview(person_name, person_email, date, time)

        if success:
            st.sidebar.success("Interview Scheduled Successfully!")
        else:
            st.sidebar.error("Failed to Schedule Interview")
