import streamlit as st
from utils.constants import *
import torch
from llama_index import (GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, 
                         ServiceContext, LangchainEmbedding)
from langchain.embeddings import HuggingFaceInstructEmbeddings
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

st.title("💬 Chat with My AI Assistant")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
local_css("styles/styles_chat.css")

# Get the variables from constants.py
pronoun = info['Pronoun']
name = info['Name']
subject = info['Subject']
full_name = info['Full_Name']

# Initialize the chat history
if "messages" not in st.session_state:
    welcome_msg = f"Hi! I'm {name}'s AI Assistant, Buddy. How may I assist you today?"
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

# App sidebar
with st.sidebar:
    st.markdown("""
                # Chat with my AI assistant
                """)
    with st.expander("Click here to see FAQs"):
        st.info(
            f"""
            - What are {pronoun} strengths and weaknesses?
            - What is {pronoun} expected salary?
            - What is {pronoun} latest project?
            - When can {subject} start to work?
            - Tell me about {pronoun} professional background
            - What is {pronoun} skillset?
            - What is {pronoun} contact?
            - What are {pronoun} achievements?
            """
        )
    
    import json
    messages = st.session_state.messages
    if messages is not None:
        st.download_button(
            label="Download Chat",
            data=json.dumps(messages),
            file_name='chat.json',
            mime='json',
        )

    st.caption(f"© Made by {full_name} 2023. All rights reserved.")


with st.spinner("Initiating the AI assistant. Please hold..."):
    # Check for GPU availability and set the appropriate device for computation.
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Global variables
    llm_hub = None
    embeddings = None
    
    Watsonx_API = "Watsonx_API"
    Project_id= "Project_id"

    # Function to initialize the language model and its embeddings
    def init_llm():
        global llm_hub, embeddings
        
        params = {
            GenParams.MAX_NEW_TOKENS: 512, # The maximum number of tokens that the model can generate in a single run.
            GenParams.MIN_NEW_TOKENS: 1,   # The minimum number of tokens that the model should generate in a single run.
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE, # The method used by the model for decoding/generating new tokens. In this case, it uses the sampling method.
            GenParams.TEMPERATURE: 0.7,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
            GenParams.TOP_K: 50,          # The top K parameter restricts the token generation to the K most likely tokens at each step, which can help to focus the generation and avoid irrelevant tokens.
            GenParams.TOP_P: 1            # The top P parameter, also known as nucleus sampling, restricts the token generation to a subset of tokens that have a cumulative probability of at most P, helping to balance between diversity and quality of the generated text.
        }
        
        credentials = {
            'url': "https://us-south.ml.cloud.ibm.com",
            'apikey' : Watsonx_API
        }
    
        model_id = ModelTypes.LLAMA_2_70B_CHAT
        
        model = Model(
            model_id= model_id,
            credentials=credentials,
            params=params,
            project_id=Project_id)
    
        llm_hub = WatsonxLLM(model=model)
    
        #Initialize embeddings using a pre-trained model to represent the text data.
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
        )
    
    init_llm()
    
    # load the file
    documents = SimpleDirectoryReader(input_files=["bio.txt"]).load_data()
    
    # LLMPredictor: to generate the text response (Completion)
    llm_predictor = LLMPredictor(
            llm=llm_hub
    )
                                    
    # Hugging Face models can be supported by using LangchainEmbedding to convert text to embedding vector	
    embed_model = LangchainEmbedding(embeddings)
    
    # ServiceContext: to encapsulate the resources used to create indexes and run queries    
    service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, 
            embed_model=embed_model
    )      
    # build index
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

def ask_bot(user_query):

    global index

    PROMPT_QUESTION = """You are Buddy, an AI assistant dedicated to assisting {name} in {pronoun} job search by providing recruiters with relevant information about {pronoun} qualifications and achievements. 
    Your goal is to support {name} in presenting {pronoun}self effectively to potential employers and promoting {pronoun} candidacy for job opportunities.
    If you do not know the answer, politely admit it and let recruiters know how to contact {name} to get more information directly from {pronoun}. 
    Don't put "Buddy" or a breakline in the front of your answer.
    Human: {input}
    """
    
    # query LlamaIndex and LLAMA_2_70B_CHAT for the AI's response
    output = index.as_query_engine().query(PROMPT_QUESTION.format(name=name, pronoun=pronoun, input=user_query))
    return output

# After the user enters a message, append that message to the message history
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Iterate through the message history and display each message
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = ask_bot(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

# Suggested questions
questions = [
    f'What are {pronoun} strengths and weaknesses?',
    f'What is {pronoun} latest project?',
    f'When can {subject} start to work?'
]

def send_button_ques(question):
    st.session_state.disabled = True
    response = ask_bot(question)
    st.session_state.messages.append({"role": "user", "content": question}) # display the user's message first
    st.session_state.messages.append({"role": "assistant", "content": response.response}) # display the AI message afterwards
    
if 'button_question' not in st.session_state:
    st.session_state['button_question'] = ""
if 'disabled' not in st.session_state:
    st.session_state['disabled'] = False
    
if st.session_state['disabled']==False: 
    for n, msg in enumerate(st.session_state.messages):
        # Render suggested question buttons
        buttons = st.container()
        if n == 0:
            for q in questions:
                button_ques = buttons.button(label=q, on_click=send_button_ques, args=[q], disabled=st.session_state.disabled)


