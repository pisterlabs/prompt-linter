import streamlit as st
from streamlit_chat import message
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from helpers.vidhelper import streamlit_hide

import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_character_count(string):
    return len(re.sub(" ", "", string))


# setup streamlit page
st.set_page_config(page_title="Goldfish Bot", page_icon="")

streamlit_hide()

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=128,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

llm = ChatVertexAI(temperature=0.5, max_output_tokens=128)
context = "You are a helpful assistant"

messages = [SystemMessage(content=context)]

st.title("GoldFish Bot :tropical_fish:")
st.markdown(
    """ 
        > :black[**A Chatbot that remembers NOTHING, but calculates the COST of conversation**]
        """
)


# st.write(f"Chatbot Context: {context}")
user_input = st.text_input(
    "Your message: ",
    key="user_input",
    placeholder="Ask me anything ...",
)

# handle user input
if user_input:
    logger.info(f"G-Bot User Input: {user_input}")
    message(user_input, is_user=True)
    messages.append(HumanMessage(content=user_input))
    with st.spinner("Thinking.."):
        response = llm.predict_messages(messages)
        messages.append(AIMessage(content=response.content))
        message(response.content, is_user=False)
        logger.info(f"G-Bot AI Message: {response.content}")

        total_tokens = llm.get_num_tokens_from_messages(messages)

        in_cnt = get_character_count(user_input)
        out_cnt = get_character_count(response.content)
        cntxt_cnt = get_character_count(context)
        total_char = in_cnt + out_cnt + cntxt_cnt
        cost = (total_char * 0.0005) / 1000
        # st.write(f"Msg:{messages}")

        st.write(
            f"Langchain token count {total_tokens}, Python character count {total_char}"
        )

        st.write(f"Conversation cost: {cost}$")
        logger.info(
            f"Langchain token count {total_tokens}, Python character count {total_char},Conversation cost: {cost}$"
        )
