from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
import streamlit as st

from llmvdb import Llmvdb
from llmvdb.embedding.openai import OpenAIEmbedding
from llmvdb.llm.langchain import LangChain


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Initialize your_llm only once
if "your_llm" not in st.session_state:
    stream_handler = StreamHandler(st.empty())

    embedding = OpenAIEmbedding()
    llm = LangChain(
        instruction="너는 챗봇이야. 공감을 잘해주고 친절하게 대해줘.",
        callbacks=[stream_handler],
    )
    st.session_state["your_llm"] = Llmvdb(
        embedding,
        llm,
        file_path="data/generated_data.json",
        workspace="workspace_path",
        verbose=False,
    )

    st.session_state["your_llm"].initialize_db()


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="안녕하세요! 저는 각종 질문에 답해주는 국민 비서 조아용이에용")
    ]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = st.session_state["your_llm"].llm.set_callbacks([stream_handler])

        response = st.session_state["your_llm"].generate_response(prompt)
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response)
        )

# streamlit run demo.py
