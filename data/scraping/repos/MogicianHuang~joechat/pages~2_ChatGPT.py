import streamlit as st
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
# Chat specific components
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
st.set_page_config(page_title="ChatGPT", page_icon="🤖")

# template & prompt settings
template = """
You are a chatbot that is helpful.
Your goal is to help the user with authentic information.
{chat_history}
Human: {human_input}
Chatbot:"""
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

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

if 'llm_chain' not in st.session_state or st.session_state.ChangeModel:
    st.session_state.llm_chain = LLMChain(
        llm=ChatOpenAI(model=model,streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),openai_api_key=st.session_state["openai_api_key"], openai_api_base=st.session_state["openai_api_base"]), 
        prompt=prompt,
        verbose=True, 
        memory=memory,
    )
    st.session_state.ChangeModel = False

if 'history' not in st.session_state:
    st.session_state.history = []




for i, (query, response) in enumerate(st.session_state.history):
    with st.chat_message(name="user", avatar="user"):
        st.markdown(query)
    with st.chat_message(name="assistant", avatar="assistant"):
        st.markdown(response)
with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

prompt_text = st.chat_input(f"向{model}进行提问")

#button = st.button("发送", key="predict")

if prompt_text:
    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    res = st.session_state.llm_chain(prompt_text)
    message_placeholder.markdown(res['text'])
    st.session_state.history.append((prompt_text,res["text"]))


clear_history = st.button("🧹", key="clear_history")
if clear_history:
    st.session_state.history.clear()
    memory.clear()

