# import langchain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, SystemMessage, HumanMessage
import streamlit as st

# langchain.verbose = False


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(
            content="""
        You are a helpful video game nerd who is obsesssed with all things Nintendo. 
        You use geeky slang. You are super motivational. 
        You will help players who are stuck in their video games by giving hints first, then the answer.
    """
        ),
        AIMessage(content="What is up my dude?"),
    ]


for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, HumanMessage):
        role = "user"
    st.chat_message(role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        chat = ChatOpenAI(streaming=True)
        response = chat(messages=st.session_state.messages, callbacks=[st_callback])
        st.session_state.messages.append(AIMessage(content=response.content))
