import streamlit as st

from dotenv import load_dotenv

from loguru import logger

from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents import load_tools
from langchain.callbacks import StreamlitCallbackHandler, StdOutCallbackHandler, FileCallbackHandler


load_dotenv()

logfile = "output.log"

logger.add(logfile, colorize=True, enqueue=True)
file_callback = FileCallbackHandler(logfile)

stdout_callback = StdOutCallbackHandler()

DEFAULT_LLM_TEMPERATURE = 0.0


def setup_llm_chains():
    llm = ChatOpenAI(temperature=st.session_state.llm_temperature)


def create_state(name, start_value):
    if name not in st.session_state:
        st.session_state[name] = start_value


create_state("llm_temperature", DEFAULT_LLM_TEMPERATURE)

### start page
st.title("🤖🎓 Basic Research Agent")
st.caption("Plan-and-Execute Agent with Access to DuckDuckGo, arXiv, PubMed and Wikipedia")

llm = ChatOpenAI(temperature=0, streaming=True, verbose=True)
tools = load_tools(["ddg-search", "arxiv", "pubmed", "wikipedia"])
planner = load_chat_planner(llm)
planner.llm_chain.verbose = True
executor = load_agent_executor(llm, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback, file_callback, stdout_callback])
        st.write(response)
