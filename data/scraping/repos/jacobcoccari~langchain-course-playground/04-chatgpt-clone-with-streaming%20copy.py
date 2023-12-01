import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler


load_dotenv()
model = ChatOpenAI(
    streaming=True,
)
memory = ConversationBufferMemory()


class StreamlitCallbacks(BaseCallbackHandler):
    # This tells the method that we will call it every time the LLM returns us a new token.
    def __init__(self, prompt) -> None:
        self.prompt = prompt
        self.full_response = ""
        self.container = st.chat_message("assistant")
        self.message_placeholder = st.empty()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.full_response += token
        self.message_placeholder.markdown(self.full_response + "▌")

    def on_chain_start(self, serialized, input_str, **kwargs):
        generate_assistant_response(self.prompt)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": input_str["input"],
            }
        )
        with st.chat_message("user"):
            st.markdown(input_str["input"])

    def on_chain_end(self, outputs, **kwargs):
        with st.chat_message("user"):
            st.markdown(outputs["response"])
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": outputs["response"],
            }
        )


def generate_assistant_response(prompt):
    chain = ConversationChain(
        llm=model,
        memory=memory,
    )
    callback = StreamlitCallbacks(prompt)
    chain.run(prompt, callbacks=[callback])


def main():
    st.title("ChatGPT Clone with ConversationChain")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("What is up?")

    if prompt:
        generate_assistant_response(prompt)


if __name__ == "__main__":
    main()
