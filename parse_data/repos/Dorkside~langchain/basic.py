from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate

import os
import streamlit as st


from choices import CHOICES

if "langchain_api_key" in st.secrets:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = st.secrets.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = "local-chat"

openai_api_key = st.secrets.openai_api_key

st.set_page_config(page_title="Daz", page_icon="📖")
st.title("📖 Daz")

if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.7

from sidebar import Sidebar

sb = Sidebar()

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")

memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("Please tell me about yourself.")

view_messages = st.expander("View the message contents in session state")

if st.session_state.template:
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=CHOICES[st.session_state.template])
chat = ChatOpenAI(openai_api_key=openai_api_key, streaming=True, callbacks=[], temperature=st.session_state.temperature, model_name="gpt-4")
if prompt is not None:
    llm_chain = LLMChain(llm=chat, prompt=prompt, memory=memory)
else:
    llm_chain = LLMChain(llm=chat, memory=memory)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

class StreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

# If user inputs a new prompt, generate and draw a new response
if user_input := st.chat_input():
    st.chat_message("human").write(user_input)
    chat_box = st.chat_message("ai").empty()
    stream_handler = StreamHandler(chat_box, display_method='write')

    chat.callbacks = [stream_handler]

    # Note: new messages are saved to history automatically by Langchain during run
    try:
        llm_chain.run(user_input)
    except Exception as e:
        st.error(e)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Memory initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
