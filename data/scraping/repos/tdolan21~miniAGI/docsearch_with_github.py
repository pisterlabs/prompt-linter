from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
import os
import streamlit as st
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from pathlib import Path
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import PostgresChatMessageHistory
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain import OpenAI
import os
import psycopg2
import unstructured
from psycopg2 import OperationalError



load_dotenv()




st.set_page_config(page_title="PGVector Docsearch", page_icon="📚")




loader = DirectoryLoader(Path(os.getenv("DOCUMENTS_PATH")))
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()


CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER"),
    host=os.environ.get("PGVECTOR_HOST"),
    port=int(os.environ.get("PGVECTOR_PORT",)),
    database=os.environ.get("PGVECTOR_DATABASE"),
    user=os.environ.get("PGVECTOR_USER"),
    password=os.environ.get("PGVECTOR_PASSWORD")
)
COLLECTION_NAME = "documentation"


db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=os.getenv("CONNECTION_STRING"),
)
store = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=os.getenv("CONNECTION_STRING"),
    embedding_function=embeddings,
)

history = PostgresChatMessageHistory(
    connection_string=os.getenv("CONNECTION_STRING"),
    session_id=os.getenv("SESSION_ID")
)

st.subheader("PGVector Document Search")
st.info("Load files to PostgreSQL by putting them into the `documents` folder. Then, click the button below to load them into the database.")


sidebar_info = st.sidebar.info("Use this to import github repositories into the database.")
# Clone
repo_input = st.sidebar.text_input("Enter repository path:", value="/app/documents/repositories")
# Input for repository name
repo_name = st.sidebar.text_input("Enter repository link:", value="")

process_button = st.sidebar.button("Process")



# If the process button is clicked, execute the logic
if process_button:
    # Clone
    repo_path = os.path.join(repo_input, repo_name.split("/")[-1])
    repo = Repo.clone_from(repo_name, to_path=repo_path)

    # Load
    loader = GenericLoader.from_filesystem(
        repo_path+"/",
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents = loader.load()
    st.write(len(documents))

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                chunk_size=2000, 
                                                                chunk_overlap=200)
    texts = python_splitter.split_documents(documents)
    st.write(len(texts))
    


from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory

db.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25}
)
tool = create_retriever_tool(
    db.as_retriever(), 
    "search_postgres",
    "Searches and returns documents from the postgreSQL vector database."
)
tools = [tool]
llm = ChatOpenAI(model_name="gpt-4") 
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=False)
from langchain.chains.question_answering import load_qa_chain
qa_chain = load_qa_chain(OpenAI(temperature=0.4), chain_type="stuff")
qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=db.as_retriever())


agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

if "shared" not in st.session_state:
   st.session_state["shared"] = True
   
# Create a radio button with two options
# execution_mode = st.sidebar.radio("Choose Execution Mode:", ("Use QA", "Use Agent Executor"))

# Based on the selected option, execute the corresponding code
if prompt := st.chat_input():
    # if execution_mode == "Use QA":
    
    result = qa.run(prompt)  # Replace 'qa' with the appropriate function
      # Replace 'agent_executor' with the appropriate function

    st.chat_message("user").write(prompt)
    

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        st.write(prompt)

        # Conditional write based on the selected execution mode
        # if execution_mode == "Use QA":
        st.write(result)
        
        
       





