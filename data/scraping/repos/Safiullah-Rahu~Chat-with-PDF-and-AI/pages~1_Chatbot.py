# Importing the required modules
import os 
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks import get_openai_callback
import logging
import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.vectorstores import Pinecone
import pinecone 
from PIL import Image
import re
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Setting up Streamlit page configuration
st.set_page_config(
    page_title="AI Chatbot", layout="centered", initial_sidebar_state="expanded"
)


with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

@st.cache_resource
def load_avaters():
    image_human = Image.open("pages/human.png")
    image_ai = Image.open("pages/ai.png")

    return image_human, image_ai


# Getting the OpenAI API key from Streamlit Secrets
openai_api_key = st.secrets.secrets.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key

# Getting the Pinecone API key and environment from Streamlit Secrets
PINECONE_API_KEY = st.secrets.secrets.PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_ENV = st.secrets.secrets.PINECONE_ENV
os.environ["PINECONE_ENV"] = PINECONE_ENV

# Initialize Pinecone with API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

#@st.cache_data
def index_namespaces():
    pinecone_index = "aichat"
    time.sleep(5)
    if pinecone_index in pinecone.list_indexes():
        index = pinecone.Index(pinecone_index)
        index_stats_response = index.describe_index_stats()
        # Define the options for the dropdown list
        opts = list(index_stats_response['namespaces'].keys())
    return opts

@st.cache_resource
def init_memory():
    return ConversationBufferWindowMemory(
        k=3,
        memory_key='chat_history',
        #output_key="answer",
        verbose=True,
        return_messages=True)

memory = init_memory()

def chat(chat_na):
    # Set the model name and Pinecone index name
    model_name = "gpt-3.5-turbo" 
    pinecone_index = "aichat"

    # Set the text field for embeddings
    text_field = "text"

    # Create OpenAI embeddings
    embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')

    
    # load a Pinecone index
    index = pinecone.Index(pinecone_index)
    db = Pinecone(index, embeddings.embed_query, text_field, namespace=chat_na)
    retriever = db.as_retriever()
    
    # Enable GPT-4 model selection
    mod = st.sidebar.checkbox('Access GPT-4')
    if mod:
        pas = st.sidebar.text_input("Write access code", type="password")
        if pas == "ongpt":
            MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
            model_name = st.sidebar.selectbox(label="Select Model", options=MODEL_OPTIONS)

    
    # _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
    # standalone question without changing the content in given question.

    # Chat History:
    # {chat_history}
    # Follow Up Input: {question}
    # Standalone question:"""
    # condense_question_prompt_template = PromptTemplate.from_template(_template)

    # prompt_template = """You are helpful information giving QA System and make sure you don't answer anything 
    # not related to following context. You are always provide useful information & details available in the given context. Use the following pieces of context to answer the question at the end. 
    # Also check chat history if question can be answered from it or question asked about previous history. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

    # {context}
    # Chat History: {chat_history}
    # Question: {question}
    # Long detailed Answer:"""

    # qa_prompt = PromptTemplate(
    #     template=prompt_template, input_variables=["context", "chat_history","question"]
    # )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Define the conversational chat function
    chat_history = st.session_state.chat_history
    @st.cache_resource
    def conversational_chat(query):
        llm = ChatOpenAI(model=model_name)

        docs = db.similarity_search(query)
        qa = load_qa_chain(llm = llm, 
                           chain_type = "stuff",
                           #memory = memory,
                           verbose = True)
        # Run the query through the RetrievalQA model
        # result = qa.run(input_documents=docs, question=query) #chain({"question": query, "chat_history": st.session_state['history']})
        #st.session_state['chat_history'].append((query, result))#["answer"]))

        return qa, docs #["answer"]
        # #retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        # llm = ChatOpenAI(model_name = model_name, temperature=0.1)
        # question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory, verbose=True)
        # doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt, verbose=True)
        # agent = ConversationalRetrievalChain(
        #     retriever=db.as_retriever(search_kwargs={'k': 6}),
        #     question_generator=question_generator,
        #     combine_docs_chain=doc_chain,
        #     memory=memory,
        #     verbose=True,
        #     # return_source_documents=True,
        #     # get_chat_history=lambda h :h
        # )

        # return agent
    # def conversational_chat(query):
        
    #     # chain_input = {"question": query}#, "chat_history": st.session_state["history"]}
    #     # result = chain(chain_input)
    #     llm = ChatOpenAI(model=model_name)
    #     docs = db.similarity_search(query)
    #     qa = load_qa_chain(llm=llm, chain_type="stuff")
    #     # Run the query through the RetrievalQA model
    #     result = qa.run(input_documents=docs, question=query) #chain({"question": query, "chat_history": st.session_state['history']})
    #     #st.session_state['history'].append((query, result))#["answer"]))
    
    #     return result   #["answer"]
    


    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model_name

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    image_human, image_ai = load_avaters()

    # if "image_human" not in st.session_state:
    #     st.session_state.image_human = image_human
    # if "image_ai" not in st.session_state:
    #     st.session_state.image_ai = image_ai
    # st.session_state.image_ai = image_ai
    # st.session_state.image_human = image_human
    pattern = r'[A-Za-z]'  # General pattern for alphabet characters

    index_filter = None

    if prompt := st.chat_input():
        matches = re.findall(pattern, prompt)

        if len(matches) > 0:

            index_filter = {'alphabet': {"$in": matches}}
            st.sidebar.write("Pattern matches:", matches)
            st.sidebar.write("Filter:", index_filter)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content":prompt})
        # st.chat_message("user").write(prompt)
        # Display user message in chat message container
        with st.chat_message("human", avatar="https://raw.githubusercontent.com/Safiullah-Rahu/Chat-with-PDF-and-AI/main/pages/human.png" ):
            st.markdown(prompt)
        with st.chat_message("ai", avatar="https://raw.githubusercontent.com/Safiullah-Rahu/Chat-with-PDF-and-AI/main/pages/ai.png" ):
            message_placeholder = st.empty()
            agent, docs = conversational_chat(prompt)
            st_callback = StreamlitCallbackHandler(st.container())
            with st.spinner("Thinking..."):
                with get_openai_callback() as cb:
                    response = agent.run(input_documents=docs, question=prompt)#agent({'question': prompt, 'chat_history': st.session_state.chat_history})#, callbacks=[st_callback])
                    st.session_state.chat_history.append((prompt, response + "\n\n\nErstellt mit Chatgpt Model: " + model_name))
                    #st.write(response)
                    message_placeholder.markdown(response + "\n\n\nErstellt mit Chatgpt Model: " + model_name)
                    st.session_state.messages.append({"role": "assistant", "content": response+"\n\n\nErstellt mit Chatgpt Model: " + model_name})
                st.sidebar.header(f"Total Token Usage: {cb.total_tokens}")

if authentication_status:
    authenticator.logout('Logout', 'main', key='unique_key')
    st.session_state.chat_namesp = ""
    chat_pass = st.sidebar.text_input("Enter chat password: ", type="password")
    if chat_pass == "chatme":
        options = index_namespaces()
        # pinecone_index = "aichat"
        # time.sleep(5)
        # if pinecone_index in pinecone.list_indexes():
        #     index = pinecone.Index(pinecone_index)
        #     index_stats_response = index.describe_index_stats()
        #     # Define the options for the dropdown list
        #     options = list(index_stats_response['namespaces'].keys())

        pri_na = st.sidebar.checkbox("Access Private Namespaces")
        chat_namespace = None

        # Check if private namespaces option is selected
        if pri_na:
            pri_pass = st.sidebar.text_input("Write access code:", type="password")
            if pri_pass == "myns":
                #st.sidebar.write("Namespaces:👇")
                #st.sidebar.write(options)
                # Create a dropdown list
                chat_namespace = st.sidebar.selectbox(label="Select Namespace", options = options)
                #chat_namespace = st.sidebar.text_input("Enter Namespace Name: ")
                st.session_state.chat_namesp = chat_namespace
            else:
                st.info("Enter the correct access code to use private namespaces!")
        else:
            # Filter the options to exclude strings ending with ".sec"
            filtered_list = [string for string in options if not string.endswith(".sec")]
            # st.sidebar.write("Namespaces:👇")
            # st.sidebar.write(filtered_list)
            chat_namespace = st.sidebar.selectbox(label="Select Namespace", options = filtered_list)
            # chat_namespace = st.sidebar.text_input("Enter Namespace Name: ")
            st.session_state.chat_namesp = chat_namespace

        chat_na = st.session_state.chat_namesp
        st.write(f"Selected Namespace Name: {chat_na}")
        # Define a dictionary with namespaces and their corresponding messages
        option_messages = {
            "test-1": "This is the message for test-1",
            "test-2": "This is the message for test-2",
            "test-3.sec": "This is the message for test-3.sec"
        }
        selected_option = list(option_messages.keys())
        # Check if the selected option is present in the dictionary
        if chat_na in selected_option:
            # Get the corresponding message for the selected option
            message_ = option_messages[chat_na]
            # Display the message
            st.write("Message:", message_)
        else:
            # If the selected option is not found in the dictionary, display a default message
            st.write("No message found for the selected option")
        chat(chat_na)

elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
