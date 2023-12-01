# from collections import namedtuple
# import altair as alt
# import math
# import pandas as pd
import streamlit as st

import openai

from modules.state import read_url_param_values, set_url_param_value
from functools import partial

from Home import APP_TITLE, APP_ICON


st.set_page_config(
    page_title=f"{APP_TITLE} - OpenAI Query",
    page_icon=APP_ICON
)

config = read_url_param_values()

DEFAULT_CONFIG = {
    "temperature": config["temperature"],
    "max_tokens": config["max_tokens"],
    "top_p": config["top_p"],
}

def openai_call(inputdata, system, config=DEFAULT_CONFIG, raw=False):
    # openai.api_key = api_key

    # Calls openai API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": inputdata},
        ],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
    )
    if raw:
        return response
    return response.get("choices")[0]["message"]["content"]

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

with st.echo(code_location='below'):
    api_key = config["api_key"]
    text_input = st.text_area("Question for gpt")

    if st.button("Run query"):
        with st.spinner("Querying the bot..."):
            openai.api_key=api_key
            resp = openai_call(text_input, "You are assisting the user. You can use Markdown for Rich format.")
            st.markdown(resp, unsafe_allow_html=True)


