import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory,ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from streamlit_chat import message
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from htmlTemplates import css, bot_template, user_template    
    
    
load_dotenv()#load your env file

st.set_page_config(page_title="MultiDocWebChat",
                       page_icon=":sunglasses:")

st.write(css, unsafe_allow_html=True)

if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

if "agent" not in st.session_state:
        st.session_state.agent = None

if 'generated' not in st.session_state: 
        st.session_state['generated'] = []

if 'past' not in st.session_state: 
        st.session_state['past'] = []




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
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print(vectorstore)
    return vectorstore

def get_conversation_chain_memory(vectorstore):

    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    search = SerpAPIWrapper() ##enabling the chatbot to access internet for questions unrelated to the document.
    tools = [
        Tool(
            name = "Question answering from the given document",
            func=conversation_chain.run,
            description="use this tool when you need to answer questions based on the context and memory available"
        ),

        Tool(
            name = "Current Search",
            func=search.run,
            description="use this tool when you need to answer questions about current events or the current state of the world"
        )
     ]
    
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=memory,
        handle_parsing_errors=True
          )
    return agent


def handle_userinput(user_question):
    try:
        print("entering pdf")
        response_agent = st.session_state.agent(user_question)
        st.session_state.chat_history = response_agent['chat_history'] 
        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(msg.content, is_user=True, key=str(i) + '_user')
            else:
                message(msg.content, is_user=False, key=str(i) + '_ai')
    except:
         response_agent="Sorry I am unable to answer your question because you haven't pressed the Process button.Click on the button after uploading the documents to continue further"
         #st.write(response_agent)
         message(response_agent, is_user=False, key=str(response_agent) + '_ai')
        


llm = ChatOpenAI()

memory_search= ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
search = SerpAPIWrapper()
tools = [

        Tool(
        name = "Current Search",
        func=search.run,
        description="use this tool when you need to answer questions about current events or the current state of the world"
        )
]

# Initialize agent
agent_search = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=memory_search,
        handle_parsing_errors=True
)



def handle_search(user_question):
     print("entering search")
     response_search = agent_search.run(user_question)
     st.session_state.past.append(user_question)
     st.session_state.generated.append(response_search)
     for i in range(len(st.session_state.past)):
                st.write(user_template.replace("{{MSG}}",st.session_state.past[i] ), unsafe_allow_html=True)
                st.write(bot_template.replace("{{MSG}}",st.session_state.generated[i] ), unsafe_allow_html=True)




st.header("Chat with your PDF'S:books:")
user_question =st.chat_input("Ask a question about your documents:")

    

with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your PDF's here and click on 'Process'", accept_multiple_files=True)
    ask_button=st.button("Process") 
    if  ask_button :
            with st.spinner("Processing"):
            #get pdf text
                raw_text=get_pdf_text(pdf_docs)

                #get the text chunks
                text_chunks = get_text_chunks(raw_text)

                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                st.session_state.agent = get_conversation_chain_memory(
                            vectorstore)
    
if user_question:
     if pdf_docs ==[]  :
          handle_search(user_question)
     else:
          handle_userinput(user_question)
          
 

