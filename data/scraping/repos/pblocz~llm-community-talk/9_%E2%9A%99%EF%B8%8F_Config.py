import streamlit as st

from Home import APP_TITLE, APP_ICON
import openai


st.set_page_config(
    page_title=f"{APP_TITLE} - Config",
    page_icon=APP_ICON
)

"""
# Configuration

This is a configuration page with all the settings used globally.

It is "stored" in the url as parameters, so save the url to persist the parameters.

"""

from modules.state import read_url_param_values, set_url_param_value
from functools import partial

config = read_url_param_values()

for k, v in config.items():
    if k == "model":
        continue
    st.text_input(k, value=v, key=k, on_change=partial(set_url_param_value, key=k))

# models = list(sorted([m["id"] for m in openai.Model.list()["data"]]))
models = ["gpt-3.5-turbo", "gpt-4"]
default_model = models.index(config.get("model") or "gpt-3.5-turbo")

selected_model = st.selectbox("OpenAI Model", models, default_model, key="model", on_change=partial(set_url_param_value, key="model"))
print(selected_model)
