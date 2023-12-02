import os
import streamlit as st
import openai

from dotenv import load_dotenv
from streamlit_chat import message
from langchain.agents import (
    initialize_agent
)
from langchain.chat_models import ChatOpenAI
from langchain.tools.base import (
    BaseTool,
)
from langchain import GoogleSearchAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    HumanMessage,
    AIMessage
)

from pathlib import Path
from llama_index import download_loader

from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatOpenAI(temperature=0, model_name="gpt-4")
search = GoogleSearchAPIWrapper(google_api_key = google_api_key)
index_directory = "./index_context"

DocxReader = download_loader("DocxReader")

loader = DocxReader()

# セッション内に保存されたチャット履歴のメモリの取得
try:
    memory = st.session_state["memory"]
except:
    memory = ConversationBufferMemory(return_messages=True)

# チャット履歴（HumanMessageやAIMessageなど）を格納する配列の初期化
history = []

class CustomSearchTool(BaseTool):
    name = "Search"
    description = "useful for when you need to answer questions about current events"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return search.run(query)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

# 画面部分
st.title("LlamaIndex + LangChain + Word + GPT4 AI for GitHub in Streamlit")
st.caption("by Marthur")

target_file_path = st.text_input("対象ファイルパス")
file_read_button = st.button("ローカルファイル読み込み")
user_input = st.text_input("質問")
send_button = st.button("送信")

# ローカルファイル読み込みボタン押下処理
if file_read_button:
    file_read_button = False

    docs = loader.load_data(file=Path(target_file_path))

    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor
    )

    tools = [
        CustomSearchTool(),
    ]

    if not os.path.exists(index_directory + "/repository"):
        repository_index = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context)
        repository_index.storage_context.persist(index_directory + "/repository")
    else:
        storage_context_repo = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=index_directory + "/repository"),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=index_directory + "/repository"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=index_directory + "/repository"),
        )
        repository_index = load_index_from_storage(storage_context_repo, service_context=service_context)
    repository_query_engine = repository_index.as_query_engine(service_context=service_context)

    class RepositoryClass(BaseTool):
        name="Repository"
        description = target_file_path + "内に存在するmemoQのプラグインについて情報の取得、表示するために使用します。"

        def _run(self, query: str) -> str:
            """Use the tool."""
            return repository_query_engine.query(query).response
    
        async def _arun(self, query: str) -> str:
            """Use the tool asynchronously."""
            raise NotImplementedError("BingSearchRun does not support async")
    tools.append(RepositoryClass())

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=True)
    agent.save_agent(index_directory + "/agent.json")
    st.session_state["agent"] = agent

    if agent: 
        memory.chat_memory.add_ai_message("読み込みました")

        # チャット履歴（HumanMessageやAIMessageなど）の読み込み
        try:
            history = memory.load_memory_variables({})["history"]
        except Exception as e:
            st.error(e)

if send_button :
    send_button = False
    agent = st.session_state["agent"]

    memory.chat_memory.add_user_message(user_input)

    response = agent.run(user_input)
    response = response.replace("mermaid", "")

    memory.chat_memory.add_ai_message(response)
    st.session_state["memory"] = memory

    # チャット履歴（HumanMessageやAIMessageなど）の読み込み
    try:
        history = memory.load_memory_variables({})["history"]
    except Exception as e:
        st.error(e)

# チャット履歴の表示
for index, chat_message in enumerate(reversed(history)):
    if isinstance(chat_message, HumanMessage):
        message(chat_message.content, is_user=True, key=2 * index)
    elif isinstance(chat_message, AIMessage):
        message(chat_message.content, is_user=False, key=2 * index + 1)
