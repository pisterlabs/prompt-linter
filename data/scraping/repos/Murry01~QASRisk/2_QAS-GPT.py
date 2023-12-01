import streamlit as st
from streamlit_option_menu import option_menu
import openai
import time

# Set openAi key
openai.api_key = st.secrets["api_secret"]


with st.sidebar:
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key")
    "[View the source code](https://github.com/Murry01/QASTRisk)"
    "[Contact](https://cmvi.knu.ac.kr/)"


st.title("QAS-GPT")
st.write("##### :red[A Question-Answering System Powered by OpenAI's GPT-3 Model]")

with st.form("my_form", clear_on_submit=False):
    a, b = st.columns([4, 1])
    search_box = a.text_input(
        label=":red[Ask QASGPT questions]",
        placeholder="Ask me any questions...",
        label_visibility="collapsed",
    )
    # b.form_submit_button("Ask Question", use_container_width=True, type="primary")
    search_button = b.form_submit_button(
        "Ask Question", use_container_width=True, type="primary"
    )


# search_box = st.text_input(
#     label=":red[Ask QASGPT questions]", placeholder="Ask me any questions..."
# )

# search_button = st.button(label="Ask question!", type="primary")

with st.spinner(text="Retrieving answer...."):
    if search_button:
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Give comprehensive answer to questions :" + search_box,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7,
        )
        res = completions.choices[0].text

        st.info(res)
    else:
        st.warning("Please enter a question!")
