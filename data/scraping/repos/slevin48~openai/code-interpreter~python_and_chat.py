from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

st.set_page_config(page_title="Python and Chat",page_icon="🐍")

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

st.sidebar.title("🐍 Python and Chat 🐱")
st.sidebar.write("*Example:*")
st.sidebar.write("What is the 10th fibonacci number?")
st.sidebar.write("""Understand, write a single neuron neural network in PyTorch.
Take synthetic data for y=2x. Train for 1000 epochs and print every 100 epochs.
Return prediction for x = 5""")

# st.sidebar.write(st.session_state.messages)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"],avatar=avatar[msg["role"]]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user",avatar=avatar["user"]).write(prompt)
    with st.chat_message("assistant",avatar=avatar["assistant"]):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = agent_executor.run(prompt, callbacks=[st_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)