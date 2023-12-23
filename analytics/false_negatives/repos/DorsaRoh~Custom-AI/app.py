"""
This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
The code creates a web application using Streamlit, a Python library for building interactive web apps.
# Author: Dorsa Rohani
# Date: AUgust 04, 2023
"""


# Import necessary libraries
import os 
import openai

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

from pdfreader import PDFDocument, SimplePDFViewer



# Set Streamlit page configuration & LOGO
from PIL import Image
# Loading Image using PIL
im = Image.open('content/logo.png')
# Adding Image to web app
st.set_page_config(page_title='CustomAI', layout='wide', page_icon = im)


# Side bar

# Side bar api key
placeholder_text_prompt = "Please enter your key"
openai_api_key = st.sidebar.text_input("OpenAI API Key",value="", help="", key="api_input", placeholder=placeholder_text_prompt)

os.environ['OPENAI_API_KEY'] = openai_api_key

st.write("#")
st.sidebar.title(":blue[CustomAI]")
st.sidebar.markdown("Train AI with custom data, revolutionizing personalized AI. Here are sample use cases:")

# CustomAI
st.sidebar.subheader(":blue[1. [RealizeAI](https://github.com/DorsaRoh/Custom-AI/tree/main/RealizeAI)]")
st.sidebar.markdown("*Think your unique knowledge has no real-world value? With RealizeAI, turn even the most obscure ideas into actionable tasks.*")
# PatientGPT.AI
st.sidebar.subheader(":blue[2. [PatientGPT.AI](https://github.com/DorsaRoh/Custom-AI/tree/main/PatientGPT.AI)]")
st.sidebar.markdown("*Doctors upload patient data, and CustomAI determines a diagnosis and specialized insight — each one crafted uniquely, for whichever disease(s) the doctor chooses.*")



# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False


# APP LAYOUT
# Title
st.title('CustomAI.')
st.subheader(':blue[Build AI trained on *your* custom data.]')
st.markdown("___")

#columns for layout
col1, col2 = st.columns(2)


 #If invalid/no api key enteblue, show warning
def valid_apikey():
    if openai_api_key.startswith('sk-'):
        return True
    else:
        st.warning('Invalid API Key', icon='⚠')
        return False
    
if valid_apikey():
    col2.markdown(":blue[AI Response:]")

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False


# Langchain LLM that feeds off of user data
def load_model():
    if PERSIST and os.path.exists("persist"):
        st.write("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    return chain

chain = load_model()
chat_history = []



# FILE SAVE TO DATA FOLDER

# Extract text from user uploaded pdf file - so LLM can read it
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as fd:
        viewer = SimplePDFViewer(fd)
        text = ""
        for page in viewer:
            viewer.render()
            text += ' '.join(viewer.canvas.strings)
    return text

def fileSaver():
    uploaded_file = st.file_uploader("Upload your data", type='.pdf')

    if uploaded_file is not None and valid_apikey():
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File has been saved successfully!")

        # create a text file with the pdf's text content
        file_path = os.path.join("data", uploaded_file.name)
        text_content = extract_text_from_pdf(file_path)
        with open(os.path.splitext(file_path)[0]+".txt", "w") as f: 
            f.write(text_content)



# Prompt templates for Langchain
class PromptTemplate:
    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        # Check if all necessary variables have been provided
        for variable in self.input_variables:
            if variable not in kwargs:
                raise ValueError(f"Missing input variable: {variable}")

        # Use the provided variables to format the template
        return self.template.format(**kwargs)

title_template = PromptTemplate(
    input_variables=['topic'], 
    template='Provide a detailed and clear and aesthetic of how can one apply the knowledge of {topic} in a real-life context and world to yield good results in money, human advancement, personal happiness, and other beneficial factors? Are there any potential applications, especially considering realistic constraints?'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'], 
    template=(
        "Present a thorough, well-articulated, and aesthetically appealing guide on the practical application of {title} in real-world scenarios. How might leveraging insights from this topic lead to tangible benefits, such as financial prosperity, forward strides in human development, heightened personal satisfaction, and other advantageous outcomes? In this exploration, are there specific applications that stand out, especially when taking into account practical and realistic limitations or challenges? Leverage {wikipedia_research}"
    )
)




# ADDITIONAL QUESTIONS

def generate_questions_response(input_text):
    chain = load_model()
    chat_history = []

    query = input_text

    if query not in ['', 'quit', 'q', 'exit']:
        result = chain({"question": query, "chat_history": chat_history})
        st.write(f"AI: {result['answer']}")
        chat_history.append((query, result['answer']))

#LAYOUT FOR COLUMN 1
with col1:
    chain = load_model()

    placeholder_text_prompt = "Analyze my sales data and predict my next month's revenue"
    script = st.text_input("Enter Prompt:",value="", help="", key="prompt_input", placeholder=placeholder_text_prompt)
    
    fileSaver()

    with st.form('additional_questions_form'):
        placeholder_text_additional = "In what industries will our product have the most transformative impact?"
        query = st.text_area('Enter additional questions:',value="", help="", key="additional_input", placeholder=placeholder_text_additional)    
        submitted = st.form_submit_button(label='Submit')
        valid_apikey()
        if submitted and valid_apikey():
            generate_questions_response(query)
    

wiki = WikipediaAPIWrapper()


if script and valid_apikey():
    try:
        title_prompt = title_template.format(topic=script)
        title_result = chain({"question": title_prompt, "chat_history": chat_history})
        chat_history.append((title_prompt, title_result['answer']))

        wiki = WikipediaAPIWrapper()
        wiki_research = wiki.run(script) 

        script_prompt = script_template.format(title=script, wikipedia_research=wiki_research)

        
        script_result = chain({"question": script_prompt, "chat_history": chat_history})
        with col2:
            st.write(f"AI: {script_result['answer']}")
            
            chat_history.append((script_prompt, script_result['answer']))
    except TypeError as e:
        st.write("An error occurred: ", e)



# Hide Streamlit's default footer
st.markdown('<style>footer{visibility:hidden;}</style>', unsafe_allow_html=True)