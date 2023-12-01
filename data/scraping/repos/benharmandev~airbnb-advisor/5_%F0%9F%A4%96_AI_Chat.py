import webbrowser
from pathlib import Path

import streamlit as st
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase

import constants

st.set_page_config(
    page_title="Airbnb Advisor | AI Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Configure the sidebar
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.markdown("Developed by Ben Harman and powered by Streamlit.")
if st.sidebar.button("🌐 benharman.dev"):
    webbrowser.open_new_tab("https://benharman.dev")

st.title("🤖 Airbnb Advisor | AI Chat")

st.write(
    """
    Welcome to the Airbnb Advisor! I'm here to assist you with insights from our Airbnb listings database.
    
    **Note:** This bot is specialized in answering questions about the dataset and cannot answer general questions.

    **Example Queries:**
    """
)

example_queries = [
    "How many listings are in San Diego?",
    "What percent of listings are owned by superhosts?",
    "What's the average review score of listings in Los Angeles?",
]

for query in example_queries:
    if st.button(query, type="primary"):
        user_query = query
        # Now, the user_query will be processed the same way as it's processed below in the code.
        # The code to process the user_query can be put into a separate function to avoid duplication.
        # For simplicity, I'm just setting user_query here.

# Database URI from constants
db_uri = f"sqlite:///{constants.DATABASE_PATH}"

# Set or get user's OpenAI API key
if "openai_query_count" not in st.session_state:
    st.session_state.openai_query_count = 0

if st.session_state.openai_query_count < 5:
    openai_api_key = st.secrets["openai_key"]
else:
    if "user_openai_key" not in st.session_state:
        openai_api_key_input = st.sidebar.text_input(
            label="OpenAI API Key (you've exceeded the 5 query limit using the default key)",
            type="password",
        )
        if openai_api_key_input:
            st.session_state.user_openai_key = openai_api_key_input
            openai_api_key = openai_api_key_input
        else:
            st.warning("Please add your OpenAI API key to continue.")
            st.stop()
    else:
        openai_api_key = st.session_state.user_openai_key

# Setup agent
llm = OpenAI(
    openai_api_key=openai_api_key,
    temperature=0,
    streaming=True,
    model="gpt-3.5-turbo-instruct",
)


@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri)


db = configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query_input = st.chat_input(
    placeholder="Ask me anything about the Airbnb dataset!"
)

if user_query_input:
    user_query = user_query_input

    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

    st.session_state.openai_query_count += 1
