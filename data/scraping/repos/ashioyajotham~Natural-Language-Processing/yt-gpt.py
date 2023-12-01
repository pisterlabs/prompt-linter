from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Set up summarization chain
from langchain.document_loaders import YoutubeLoader, PyPDFLoader, UnstructuredFileLoader
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, LlamaCpp
from langchain import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from serpapi import GoogleSearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.agents import AgentExecutor, load_tools, ZeroShotAgent, Tool, initialize_agent, AgentType
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

import streamlit as st

llm = OpenAI(temperature=0)

st.set_page_config(
    page_title="A S H I O Y A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
choice = st.radio("", ("YouTube", "Summary query", "About"), horizontal=True)

if choice == "YouTube":
    st.title("LangChain Demo: Summarize Youtube Videos")
    st.info("This is a demo of the LangChain library. It uses the GPT-3 model to summarize Youtube videos. ")

# Add a text input
    prompt = st.text_input("Enter a Youtube URL")

    if prompt:
        loader = YoutubeLoader.from_youtube_url(prompt, add_video_info=False)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, separator=" ", chunk_overlap=50)
        split_docs = splitter.split_documents(docs)
        if split_docs:
            with st.spinner("Summarizing..."):
                chain = load_summarize_chain(llm, chain_type='map_reduce')
                summary = chain.run(split_docs)
            st.success("Done!")
            st.write(summary)
            # Automatically save the summary to a text file
            with open("summary.txt", "w") as f:
                f.write(summary)

    
        
# Save the text in a variable
#text = summary

# save the text in a string
#text = str(text)
if choice == 'Summary query':
    st.title("LangChain Demo: Summarize Youtube Videos")
    st.info("This is a demo of the LangChain library. It uses the GPT-3 model to summarize Youtube videos. ")

    ## vectorstore info
    loader = UnstructuredFileLoader("summary.txt")
    text = loader.load()
    text = loader.load_and_split()
    vectorstore_info = VectorStoreInfo(
        name="youtube",
        description="Youtube video summary",
        vectorstore=Chroma.from_documents(text, embedding=OpenAIEmbeddings(model='davinci')),
    )

    ## Alternatively
    from langchain.indexes import VectorstoreIndexCreator
    index_creator = VectorstoreIndexCreator(vectorstore_cls = Chroma,
                                            embedding = OpenAIEmbeddings(model='davinci'),
                                            text_splitter=CharacterTextSplitter(chunk_size=1000, separator=" ", chunk_overlap=0))

    
    ## toolkit
    prompt="Your name is AshioyaJ and you are an AI assistant tasked with summarizing Youtube videos. Use the Youtube video summary to answer the following questions."
    tool_names = ["serpapi"]
    tools = load_tools(tool_names)
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    
    agent_executor = create_vectorstore_agent(
        llm = llm,
        toolkit = toolkit,
        verbose=True,
    )

    query = st.text_input("Enter a query")
    response = agent_executor.run(query)
    st.write(response)

if choice == "About":

    # About on main section
    st.subheader("About")
    st.write("This is v1 (beta). It is still under development. Please report any issues on the Github repo.")

    st.subheader("How it works")
    st.write("This demo uses the GPT-3 model to summarize Youtube videos. It uses the OpenAI API to run the model. The model is then used to summarize the Youtube video. The video is split into chunks of 1000 characters. The model is run on each chunk. The chunks are then combined into a single summary.")

    st.subheader("Works in progress")
    st.markdown("""
    - [ ] Add document summarization with different models
    - [ ] Add music generation with MusicGen
    - [x] Text generation (The  results are not very good. I am working on improving them)
    """)

    st.subheader("Acknowledgements")
    st.write("This project could not have been possible without the following projects:")
    st.markdown("""
    - [PromptEngineer48](https//github.com/PromptEngineer48) for the implementation of the summarization chain. [See the full work here](https://youtube.com/watch?v=g9N8hVKPC1o)
    - [Yvann-Hub](https://github.com/yvann-hub) for the implementation of the Youtube loader. [See the full work here](https://github.com/yvann-hub/Robby-chatbot)
    """)

# Add year and copyright logo
st.sidebar.subheader("Copyright")
st.sidebar.write("© 2023 Ashioya Jotham")

# Make the sidebar slide
st.sidebar.subheader("About the author")
st.sidebar.markdown("""
- [![Github](https://img.shields.io/badge/Github-100000?style=for-the-badge&logo=github&logoColor=white)](<htpps://github.com/ashioyajotham>)
- [![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](<https://twitter.com/ashioyajotham>)
- [![Linkedin](https://img.shields.io/badge/Linkedin-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](<https://www.linkedin.com/in/ashioya-jotham-0b1b3b1b2/>)
""")