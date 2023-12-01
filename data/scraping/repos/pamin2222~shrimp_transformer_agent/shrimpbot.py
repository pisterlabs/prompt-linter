import openai
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

import shrimp_helper
from key_helper import check_openai_key
import streamlit as st

# Add input for OpenAI API key
with st.sidebar:
    target_word_input = st.text_input(
        "Text input for AI to use",
        "Shrimp",
        key="placeholder",
        max_chars=10
    )
    ai_mode_selection = st.radio(
        "Set word replace mode 👉",
        key="ai_mode",
        options=["Partial Shrimp Mode", "Full Shrimp Mode", "Normal"],
    )
    st.info("Full Shrimp Mode: Every word in the AI's response is replaced with your chosen keyword.")
    st.info("Partial Shrimp Mode: Only nouns and verbs in the AI's response are replaced with your chosen keyword.")
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Inspired by this post](https://www.facebook.com/groups/cursedaiwtf/posts/1395288517746294)"


st.title("Shrimp Transformer")
st.caption("")

# Set up memory
msgs = StreamlitChatMessageHistory(key="history")

check_openai_key(openai_api_key)

# Set up LLMs
llm_shrimp = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", request_timeout=30)
if "llm_shrimp_memory" not in st.session_state:
    llm_shrimp_memory = ConversationBufferMemory()
    st.session_state.llm_shrimp_memory = llm_shrimp_memory
else:
    llm_shrimp_memory = st.session_state.llm_shrimp_memory

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if len(msgs.messages) == 0:
    init_msg = "Input something to start a conversation"
    st.chat_message("system").write(init_msg)


def generate_conversation(latest_response, ai_mode_selection, st):
    llm_shrimp_conver_chain = ConversationChain(
        llm=llm_shrimp,
        verbose=False,
        memory=llm_shrimp_memory,
        prompt=shrimp_helper.shrimpify_prompt_template
    )

    user_w_params = shrimp_helper.create_user_input_with_params(mode=ai_mode_selection, user_prompt=latest_response,
                                                                target_word=target_word_input)

    ai_response = llm_shrimp_conver_chain.predict(input=user_w_params)
    if ai_mode_selection == "Full Shrimp Mode":
        #print("FULL_SHRIMP_MODE")
        shrimpified_response = ' '.join(
            [target_word_input if target_word_input not in word else word for word in ai_response.split()])
        #print("shrimpified_response:", shrimpified_response)
        ai_response = shrimpified_response

    # Add the AI response to the conversation container
    msgs.add_ai_message(ai_response)

    # Display the AI response in the chat interface
    st.chat_message("ai").write(ai_response)

    # The function should return the AI's response instead of the latest response from the user
    return ai_response


if prompt := st.chat_input():
    openai.api_key = openai_api_key
    msgs.add_user_message(prompt)
    st.chat_message("user").write(prompt)
    latest_response = generate_conversation(prompt, ai_mode_selection, st)
    #print("ai_mode_selection:", ai_mode_selection)
