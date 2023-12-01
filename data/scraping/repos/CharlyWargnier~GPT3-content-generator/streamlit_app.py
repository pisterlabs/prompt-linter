# ----------------------Importing libraries----------------------

import streamlit as st
import openai

# ----------------------Page config--------------------------------------

st.set_page_config(page_title="GPT3 Content Generator", page_icon="📢")

# ----------------------Tooltips--------------------------------------

toolTip0 = """

Tick this box to access advanced settings (model choice and token length)
                    
"""

toolTip1 = """

# Davinci-instruct-beta-v3
Davinci-instruct is the most capable model in the Instruct series, which is better at following instructions than the Base series. 

**Strengths:** Shorter and more naturally phrased prompts, complex intent, cause and effect.
# Curie-instruct-beta-v2
Curie-instruct is very capable but faster and lower cost than davinci-instruct. Part of the Instruct series is better at following instructions than the Base series. 

**Strengths:** Shorter and more naturally phrased prompts, language translation, complex classification, sentiment.
# Babbage-instruct-beta 
This model is part of our Instruct series, which is better at following instructions than the Base series.

"""

toolTip2 = """

**What are tokens?**

Tokens can be thought of as pieces of words. Before the API processes the prompts, the input is broken down into tokens. These tokens are not cut up exactly where the words start or end - tokens can include trailing spaces and even sub-words. Here are some helpful rules of thumb for understanding tokens in terms of lengths:

1 token ~= 4 chars in English or ¾ words

Or

1-2 sentence ~= 30 tokens

Find more [about tokens here](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)

"""

# ----------------------Custom layout width----------------------------


def _max_width_():
    max_width_str = f"max-width: 800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()

# ----------------------Sidebar section--------------------------------

st.sidebar.image(
    "logo.png",
    width=305,
)

st.sidebar.caption("")

API_Key = st.sidebar.text_input("Enter your OpenAI API key")

st.sidebar.caption("")

st.sidebar.caption(
    "Made in [Streamlit](https://www.streamlit.io/)&nbsp, with :heart: by [@DataChaz](https://twitter.com/DataChaz)"
)

# ----------------------API key section----------------------------------

if not API_Key:

    c30, c31, c32 = st.columns([1, 0.9, 3])

    st.subheader("")
    st.header("")
    st.header("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.caption("")
    st.info("First, you need to enter your OpenAI API key.")
    st.image("arrow.png", width=150)
    st.caption(
        "No OpenAI API key? Get yours [here!](https://openai.com/blog/api-no-waitlist/)"
    )

    st.stop()


c30, c31, c32 = st.columns([1, 0.9, 3])

openai.api_key = API_Key

st.title("")
st.title("")
st.header("")
st.subheader("")
st.caption("")

# ----------------------Main section--------------------------------------

checkbox_value = st.checkbox("Advanced Mode", help=toolTip0)

if checkbox_value:

    with st.form("my_form"):

        cMargin, c1, cMargin, c2, cMargin = st.columns([0.1, 2, 0.1, 2, 0.1])

        with c1:

            engine = st.selectbox(
                "Select your GPT3 engine",
                [
                    "davinci-instruct-beta-v3",
                    "curie-instruct-beta-v2",
                    "babbage-instruct-beta",
                ],
                key="3",
                help=toolTip1,
            )

            with c2:
                maxTokens = st.slider(
                    "Max tokens",
                    60,
                    2000,
                    value=1200,
                    step=100,
                    key="1",
                    help=toolTip2,
                )
        ce, c1, cf = st.columns([0.1, 5, 0.1])
        with c1:

            text_input = st.text_input(
                "What would you like to ask?",
                key="2",
                placeholder='e.g. "make a list of great French authors of the past 100 years"',
            )

        submitted = st.form_submit_button(
            "Show me the Magic! ✨", help="Generate text from your instructions"
        )


else:

    with st.form("my_form"):

        ce, c1, cf = st.columns([0.1, 5, 0.1])
        with c1:

            text_input = st.text_input(
                "What would you like to ask?",
                key="2",
                placeholder='e.g. "make a list of great French authors of the past 100 years"',
            )

        submitted = st.form_submit_button(
            "Show me the Magic! ✨", help="Generate text from your instructions"
        )

if submitted and checkbox_value:

    col1, col2, col3 = st.columns(3)
    with col2:

        gif_runner = st.image("mouse.gif")

        response = openai.Completion.create(
            engine=engine,
            max_tokens=maxTokens,
            prompt=text_input,
        )

        output_code = response["choices"][0]["text"]

    gif_runner.empty()

    output_code
    st.text("")
    st.download_button(
        "Download output",
        output_code,
        file_name="GPT_output.txt",
        help="Download the output",
    )

if submitted and not checkbox_value:

    col1, col2, col3 = st.columns(3)
    with col2:

        gif_runner = st.image("mouse.gif")

        response = openai.Completion.create(
            engine="davinci-instruct-beta-v3",
            max_tokens=1200,
            prompt=text_input,
        )

        output_code = response["choices"][0]["text"]

    gif_runner.empty()

    output_code
    st.text("")
    st.download_button(
        "Download output",
        output_code,
        file_name="GPT_output.txt",
        help="Download the output",
    )