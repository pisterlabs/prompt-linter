from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

st.set_page_config(page_title="PythonAgent",page_icon="🐍")

# Set the API key for the openai package
openai_api_key = st.secrets["OPEN_AI_KEY"]

avatar = {"assistant": "🐍", "user": "🐱"}
# Set the API key for the openai package
openai_api_key = st.secrets["OPEN_AI_KEY"]
agent_executor = create_python_agent(
    llm=ChatOpenAI(
        temperature=0, 
        model="gpt-3.5-turbo-0613", 
        openai_api_key=openai_api_key,
        streaming=True,
        ),
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    agent_executor_kwargs={"handle_parsing_errors": True},
)

st.sidebar.title("Python Agent 🐍")
st.sidebar.write("*Example:*")
st.sidebar.write("What is the 10th fibonacci number?")
st.sidebar.write("""Understand, write a single neuron neural network in PyTorch.
Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
Return prediction for x = 5""")

if prompt := st.chat_input():
    st.chat_message("user",avatar=avatar["user"]).write(prompt)
    with st.chat_message("assistant",avatar=avatar["assistant"]):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.run(prompt, callbacks=[st_callback])
        st.write(response)