"""Simple example of a chatbot (with memory) with Streamlit interface."""

# please put your API key in os.environ["OPENAI_API_KEY"]

from langchain.chains   import LLMChain
from langchain.llms     import OpenAI
from langchain.prompts  import PromptTemplate

from langchain.memory   import StreamlitChatMessageHistory
from langchain.memory   import ConversationBufferMemory


# MODEL
# =====

# prompts setup
template = """You are an AI chatbot having a conversation with a human.
Please provide responses only in JSON format.

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)


# memory setup
msgs = StreamlitChatMessageHistory(key="chat_messages")
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)

# if there are no messages in the chat history, add a welcome message
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# AI model setup
chain = LLMChain(llm=OpenAI(), prompt=prompt, memory=memory)


# INTERFACE
# =========

import streamlit as st # pylint: disable=wrong-import-position

# writes all messages to the screen
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# asks for user input
prompt = st.chat_input()

# if the user has entered something
if prompt:
    # shows the user's input on the screen
    st.chat_message("human").write(prompt)

    # generates a response from the AI
    # adds prompt and response to the chat history
    response = chain.run(prompt)

    # show the AI's response on the screen
    st.chat_message("ai").write(response)
