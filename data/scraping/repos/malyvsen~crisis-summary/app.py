from datetime import datetime
from crisis import backend as bkd, rss_sources
import streamlit as st
import openai

st.title("Crisis of Infinite Tweets!")


st.sidebar.write("Settings")
openai_key = st.sidebar.text_input("OpenAI API key", type="password")

if openai_key == "":
    st.sidebar.write("""Please enter the OpenAI key to continue!""")
    st.stop()

openai.api_key = openai_key

engines = openai.Engine.list()
engine_ids = [e["id"] for e in engines["data"]]


engine = st.sidebar.selectbox(
    "Model engine",
    options=engine_ids,
    index=(engine_ids.index("davinci") if "davinci" in engine_ids else 0),
)


temperature = st.sidebar.slider(
    "Model temperature", min_value=0.0, max_value=1.0, step=0.05, value=0.15
)
max_tokens = st.sidebar.slider(
    "Maximal answer length", min_value=1, max_value=512, value=256
)

"""Search RSS feeds and summarize info through GPT-3!"""
topic = st.selectbox("What topic interests you?", options=[r.name for r in rss_sources])


question = st.text_area(
    "What do you want to know?",
    value="What is the situation?",
)

submit_button = st.button("Submit")

if submit_button:
    f"""Submitted on {datetime.now()}"""

    response, most_important = bkd.GPT3Api(
        api_key=openai_key,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
    ).get_summarization(topic, question)

    f"""**Response**: *{response}*"""

    """**Top 3 most relevant tweets:**"""
    for t in most_important:
        t


else:
    """Press submit!"""
