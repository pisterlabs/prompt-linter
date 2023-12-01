from pathlib import Path
import streamlit as st

from langchain import SQLDatabase
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain
from langchain.llms import OpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_experimental.sql import SQLDatabaseChain

from streamlit_agent.callbacks.capturing_callback_handler import playback_callbacks
from streamlit_agent.clear_results import with_clear_container

DB_PATH = (Path(__file__).parent / "Chinook.db").absolute()

st.set_page_config(
    page_title="Merkén", page_icon="🦜", layout="wide", initial_sidebar_state="collapsed"
)

"# 🦜🔗 MERKEN"

openai_api_key = st.secrets['OPENAI_API_KEY']
enable_custom = True

# Tools setup
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, streaming=True)
search = DuckDuckGoSearchAPIWrapper()
#db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
#db_chain = SQLDatabaseChain.from_llm(llm, db)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
]

# Initialize agent
mrkl = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

with st.form(key="form"):
    user_input = st.text_input("Ask your own question")
    submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🦜")
    st_callback = StreamlitCallbackHandler(answer_container)
    answer = mrkl.run(user_input, callbacks=[st_callback])

    answer_container.write(answer)
