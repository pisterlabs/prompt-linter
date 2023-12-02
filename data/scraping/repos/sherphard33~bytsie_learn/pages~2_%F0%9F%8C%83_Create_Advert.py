
import json
import streamlit as st
import data.prompts as pr
from dotenv import load_dotenv
import utils.helper_functions as hf
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from data.htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def generate_advert_script(question_map: map):
    print(question_map)
    script_prompt = pr.advert_script_generator_prompt.format(
        product=question_map['product'],
        link=question_map['product_url'],
        desc=question_map['desc'],)
    bot_response = hf.basic_generation(script_prompt)

    return {
        "question": script_prompt,
        "bot_response": bot_response
    }


def main():
    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    bot_response = None

    st.header("Create an advert text :night_with_stars:")
    with st.sidebar:
        product = st.text_input("Enter product name:")
        product_url = st.text_input("Enter product url:")
        user_question = st.text_input("Add more context:")
        advert_map = {
            "product": product,
            "product_url": product_url,
            "desc": user_question
        }
        if st.button("Process"):
            with st.spinner("Processing"):
                bot_response = generate_advert_script(advert_map)
    if bot_response is not None:
        st.write(user_template.replace(
            "{{MSG}}", bot_response['question']), unsafe_allow_html=True)
        st.write(bot_template.replace(
            "{{MSG}}", bot_response['bot_response']), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
