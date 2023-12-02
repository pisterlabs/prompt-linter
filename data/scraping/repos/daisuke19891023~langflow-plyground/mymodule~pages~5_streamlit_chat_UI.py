import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

if "conversation" not in st.session_state:
    st.session_state["conversation"] = []


@st.cache_resource
def load_chain() -> ConversationChain:
    """Logic for loading the chain you want to use should go here."""
    # template = "{history} let's think step by step"
    # prompt = PromptTemplate(input_variables=["history"], template=template)
    chat = ChatOpenAI()
    # chain = LLMChain(llm=chat, prompt=load_translate_prompt(), verbose=True)
    chain = ConversationChain(llm=chat, memory=ConversationBufferMemory(), verbose=True)
    return chain


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Echo Bot")


st.write(st.session_state.conversation)
st.session_state.conversation = load_chain()
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        llm_response = st.session_state.conversation.run(prompt)
        st.markdown(llm_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": llm_response})
