from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
from langchain.agents import AgentType, initialize_agent, load_tools
from dotenv import load_dotenv

load_dotenv()

class QAManager:
    agent_executor = None
    chain = None
    embeddings = None
    docsearch = None

    def configure(self):
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

        # read data from the file and put them into a variable called raw_text
        loader = TextLoader("./content.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        # Download embeddings from OpenAI
        self.embeddings = OpenAIEmbeddings()

        self.docsearch = FAISS.from_documents(docs, self.embeddings)

        retriever = self.docsearch.as_retriever()
        retriever_tool = create_retriever_tool(
            retriever,
            "search_child_activities",
            "Searches and returns documents regarding the child activities and information in the kindergarten."
        )

        tools = load_tools(["openweathermap-api"], llm)

        tools.append(retriever_tool)

        system_message = SystemMessage(
            content=(
                "Do your best to answer the questions. "
                "Feel free to use any tools available to look up "
                "relevant information, only if necessary"
                "Location: Saitama, Japan"
            )
        )
        self.agent_executor = create_conversational_retrieval_agent(llm, tools, system_message=system_message,
                                                                    verbose=True)

    def run(self, query):
        answer = self.agent_executor({"input": query})
        return answer["output"]


if 'qa_manager' not in st.session_state:
    st.session_state.qa_manager = QAManager()
    st.session_state.qa_manager.configure()

# 初始化会话状态，用于存储聊天历史
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 聊天UI
st.title("子供幼稚園QA")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])


user_query = st.chat_input("質問を入力してください")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    answer = st.session_state.qa_manager.run(user_query)
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
