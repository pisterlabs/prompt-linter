from environs import Env
env = Env()
# Read .env into os.environ
env.read_env()

import streamlit as st
from streamlit_chat import message
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory
                                                  )

# Preparing the model instance

llm = AzureChatOpenAI(
    openai_api_base=env("OPENAI_API_BASE"),
    openai_api_version=env("OPENAI_API_VERSION"),
    deployment_name=env("AZURE_GPT_DEPLOYMENT"),
    openai_api_key=env("OPENAI_API_KEY"),
    openai_api_type=env("OPENAI_API_TYPE"),
)


st.set_page_config(page_title="Simple ChatBot with Memory", page_icon=":robot:")
st.header("Hey, I'm your Chat GPT with memory")


st.sidebar.header("The summary of the conversation : ")


if "human_messages" not in st.session_state:
    st.session_state.human_messages = []
    st.session_state.ai_messages = []
    st.session_state.llm_chain = ConversationChain(llm=llm,
                                verbose=True,
                                memory=ConversationSummaryMemory(llm=llm))


# Display chat messages from history on app rerun
for i in range (len(st.session_state.human_messages)):
    message(st.session_state.human_messages[i], is_user=True)
    message(st.session_state.ai_messages[i])



if user_input := st.chat_input("What is up?"):
    st.session_state.human_messages.append(user_input)

    message(user_input, is_user=True)

    ai_response = st.session_state.llm_chain.predict(input=user_input)
    st.session_state.ai_messages.append(ai_response)

    message(ai_response)

    st.sidebar.write(st.session_state.llm_chain.memory.buffer)

