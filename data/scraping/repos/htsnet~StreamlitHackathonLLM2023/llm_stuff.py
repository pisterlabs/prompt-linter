from datetime import datetime

import streamlit as st
from langchain import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith.client import Client
from streamlit_feedback import streamlit_feedback

_DEFAULT_SYSTEM_PROMPT = "You are a helpful chatbot."


def get_langsmith_client():
    return Client(
        api_key=st.session_state.langsmith_api_key,
    )


def get_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
        return_messages=True,
        memory_key="chat_history",
    )


def get_llm_chain(
    memory: ConversationBufferMemory,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
) -> LLMChain:
    """Return a basic LLMChain with memory."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt + "\nIt's currently {time}.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ],
    ).partial(time=lambda: str(datetime.now()))
    llm = ChatOpenAI(
        temperature=temperature,
        streaming=True,
        openai_api_key=st.session_state.openai_api_key,
    )
    return LLMChain(prompt=prompt, llm=llm, memory=memory or get_memory())


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def feedback_component(client):
    scores = {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0}
    if feedback := streamlit_feedback(
        feedback_type="faces",
        optional_text_label="[Optional] Please provide an explanation",
        key=f"feedback_{st.session_state.run_id}",
    ):
        score = scores[feedback["score"]]
        feedback = client.create_feedback(
            st.session_state.run_id,
            feedback["type"],
            score=score,
            comment=feedback.get("text", None),
        )
        st.session_state.feedback = {"feedback_id": str(feedback.id), "score": score}
        st.toast("Feedback recorded!", icon="📝")