# Import things that are needed generically
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
import streamlit as st
from pydantic import BaseModel, Field
import os
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import Tool
from langchain.utilities import PythonREPL
from langchain.memory import ConversationBufferMemory
from translate import Translator
st.write("## 向带有工具的ChatGPT提问")
st.write("### GPT可以更精准地回答问题")
use_chinese = st.checkbox('使用中文进行问答')
if 'ChangeModel' not in st.session_state:
    st.session_state.ChangeModel = False
def change_model():
    st.session_state.ChangeModel = True
col1 , col2 = st.columns(2)
model = col1.selectbox(
    'Model',
    ('gpt-3.5-turbo','gpt-4'),
    on_change=change_model
)
temperature = col2.slider(
    'temperature', 0.0, 1.0, 0.8, step=0.01
)

### initialize
translator1=Translator(from_lang="chinese",to_lang="english")
translator2=Translator(from_lang="english",to_lang="chinese")
python_repl = PythonREPL()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
googlesearch = GoogleSearchAPIWrapper(google_api_key=st.session_state.GOOGLE_API_KEY,google_cse_id=st.session_state.GOOGLE_CSE_ID)
llm = ChatOpenAI(temperature=0,model=model,openai_api_base=st.session_state.openai_api_base,openai_api_key=st.session_state.openai_api_key)
search = SerpAPIWrapper(serpapi_api_key=st.session_state.serpapi_api_key)
wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=st.session_state.WOLFRAM_ALPHA_APPID)
memory = ConversationBufferMemory(memory_key="chattools_history")
llm_math_chain = LLMMathChain(llm=llm, verbose=True,memory=memory)
st.session_state.inited = True
tools = [
    Tool(
        name="Google Search",
        description="Search Google for recent results.It's cheap and fast and you should use it before using the Search tool",
        func=googlesearch.run,
    ),
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
            # coroutine= ... <- you can specify an async method if desired as well
    ),
        # Tool.from_function(
        #     func=llm_math_chain.run,
        #     name="Calculator",
        #     description="useful for when you need to answer questions about math",
        #     args_schema=CalculatorInput
        #     # coroutine= ... <- you can specify an async method if desired as well
        # ),
        
    Tool(
        name="Wikipedia Search",
        description="Search Wikipedia for recent results. It's a good idea to use this tool before using the Google Search tool.",
        func=wikipedia.run,
    ),
    Tool(
        name="Wolfram Alpha",
        description="Wolfram Alpha is a computational knowledge engine. It's useful for answering questions about math, science, and more. If other tools fail, try this one.",
        func=wolfram.run,
    ),
    Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run
    )
]
llm = ChatOpenAI(temperature=0,model=model,openai_api_base=st.session_state.openai_api_base,openai_api_key=st.session_state.openai_api_key)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,handle_parsing_errors=True,
)
class CalculatorInput(BaseModel):
    question: str = Field()


if  st.session_state.ChangeModel:
    llm = ChatOpenAI(temperature=0,model=model,openai_api_base=st.session_state.openai_api_base,openai_api_key=st.session_state.openai_api_key)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True,
    )
    st.session_state.ChangeModel = False

# Load the tool configs that are needed.


# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.


if 'tool_history' not in st.session_state:
    st.session_state.tool_history = []

for i, (query, response) in enumerate(st.session_state.tool_history):
    with st.chat_message(name="user", avatar="user"):
        st.markdown(query)
    with st.chat_message(name="assistant", avatar="assistant"):
        st.markdown(response)
with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

prompt_text = st.chat_input(f"向{model}进行提问")


if prompt_text:
    input_placeholder.markdown(prompt_text)
    if use_chinese:
        prompt_text = translator1.translate(prompt_text)
    history = st.session_state.tool_history
    res = agent.run(prompt_text)
    if use_chinese:
        res = translator2.translate(res)
    message_placeholder.markdown(res)
    st.session_state.tool_history.append((prompt_text,res))


clear_history = st.button("🧹", key="clear_history")
if clear_history:
    st.session_state.tool_history.clear()
    memory.clear()