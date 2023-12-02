import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()
model = ChatOpenAI()
memory = ConversationBufferMemory(return_messages=True)


def generate_assistant_response(prompt):
    conversation = ConversationChain(
        llm=model,
        memory=memory,
    )
    response = conversation.predict(input=prompt)
    return response


def save_chat_history(prompt, messages):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    assistant_response = generate_assistant_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )


def main():
    st.title("ChatGPT Clone with ConversationChain")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("What is up?")

    if prompt:
        save_chat_history(prompt, st.session_state.messages)


if __name__ == "__main__":
    main()
