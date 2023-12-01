import os
import streamlit as st
from streamlit.logger import get_logger
# import tkinter as tk
# from tkinter import filedialog
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from chains import (
    load_llm,
    configure_llm_only_chain,
    get_qa_rag_chain
)
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.text_splitter import Language
from agent import get_agent_executor
from db import process_documents

# set page title
st.set_page_config(
    page_title="Code Explorer",
    page_icon="👨‍💻",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "GitHub: https://github.com/tobyloki/CodeExplorer"
    }
)

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

@st.cache_resource
def initLLM():
    # create llm
    llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})

    return llm

llm = initLLM()

@st.cache_resource
def get_llm_chain():
    chain = configure_llm_only_chain(llm)
    return chain

@st.cache_resource
def process_directory(language, directory, count) -> (str, Neo4jVector):
    error, vectorstore = process_documents(language, directory)
    return (error, vectorstore)

@st.cache_resource
def get_qa_chain(_vectorstore, count):
    qa = get_qa_rag_chain(_vectorstore, llm)
    return qa

@st.cache_resource
def get_agent(_qa, count):
    qa = get_agent_executor(_qa, llm)
    return qa

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # if token.endswith('?'):
        #     token += '\n\n\n'
        # token = token.replace('"', '')
        self.text += token
        self.container.markdown(self.text)

def main():
    qa = None
    agent = None
    llm_chain = get_llm_chain()

    if "language" not in st.session_state:
        st.session_state[f"language"] = None
    if "directory" not in st.session_state:
        st.session_state[f"directory"] = None
    if "detailedMode" not in st.session_state:
        st.session_state[f"detailedMode"] = True
    if "vectorstoreCount" not in st.session_state:  # only incremented to reset cache for processDocuments()
        st.session_state[f"vectorstoreCount"] = 0
    if "qaCount" not in st.session_state:           # only incremented to reset cache for get_qa_rag_chain()
        st.session_state[f"qaCount"] = 0    
    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    # # Set up tkinter
    # root = tk.Tk()
    # root.withdraw()

    # # Make folder picker dialog appear on top of other windows
    # root.wm_attributes('-topmost', 1)

    # sidebar
    with st.sidebar:
        # Convert enum values to a list of strings
        languages_list = [lang.value for lang in Language]
        default_index = languages_list.index(Language.PYTHON)
        languageSelected = st.selectbox(
            'Select language',
            languages_list,
            index=default_index
        )

        # show folder picker dialog
        # st.title('Select Folder')
        # folderClicked = st.button('Folder Picker')

        currentPath = os.getcwd()
        directory = st.text_input('Enter folder path', currentPath)
        directory = directory.strip()

        processBtnClicked = st.button('Process files')
        if processBtnClicked:
            if not os.path.exists(directory):
                st.error("Path doesn't exist!")
            else:
                # directory = filedialog.askdirectory(master=root)
                if isinstance(directory, str) and directory:
                    st.session_state[f"language"] = languageSelected
                    st.session_state[f"directory"] = directory
                    st.session_state[f"vectorstoreCount"] += 1
                    st.session_state[f"qaCount"] += 1
                    st.session_state[f"user_input"] = []
                    st.session_state[f"generated"] = []

        # show folder selected
        if st.session_state[f"directory"]:
            st.code(st.session_state[f"directory"])

            error, vectorstore = process_directory(st.session_state[f"language"], st.session_state[f"directory"], st.session_state[f"vectorstoreCount"])

            if error:
                st.error(error)
            elif vectorstore:
                qa = get_qa_chain(vectorstore, st.session_state[f"qaCount"])
                agent = get_agent(qa, st.session_state[f"qaCount"])

                # show clear chat history button
                clearMemoryClicked = st.button("🧹 Reset chat history")
                if clearMemoryClicked:
                    st.session_state[f"qaCount"] += 1
                    st.session_state[f"user_input"] = []
                    st.session_state[f"generated"] = []

                    qa = get_qa_rag_chain(vectorstore, st.session_state[f"qaCount"])
                    agent = get_agent(qa, st.session_state[f"qaCount"])

                # show toggle to switch between qa and agent mode
                detailedMode = st.toggle('Detailed mode', value=True)
                st.session_state[f"detailedMode"] = detailedMode

    # load previous chat history
    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])
        # Display all exchanges
        for i in range(0, size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])
            with st.chat_message("assistant"):
                st.write(st.session_state[f"generated"][i])

    # user chat
    user_input = st.chat_input("What coding issue can I help you resolve today?")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
            st.session_state[f"user_input"].append(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                stream_handler = StreamHandler(st.empty())
                if qa:
                    if st.session_state[f"detailedMode"]:
                        print("Using QA")
                        result = qa(
                            {"question": user_input},
                            callbacks=[stream_handler]
                        )
                        answer = result["answer"]
                    else:
                        print("Using Agent")
                        result = agent(
                            {"input": user_input},
                            callbacks=[stream_handler]
                        )
                        answer = result["output"]

                    # print("result:", result)
                else:
                    print("Using LLM only")
                    answer = llm_chain(
                        {"question": user_input},
                        callbacks=[stream_handler]
                    )

                # answer = answer.replace('"', '')
                st.session_state[f"generated"].append(answer)


if __name__ == "__main__":
    main()



