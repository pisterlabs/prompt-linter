#!/usr/bin/env python3

import json
import logging
import re
import requests

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from datetime import datetime, timedelta
from langchain.llms import OpenAI
from llama_index import GPTVectorStoreIndex, Document, LLMPredictor, ServiceContext
from requests.exceptions import HTTPError
from wordcloud import WordCloud


logger = logging.getLogger("llama_index")
logger.setLevel(logging.WARNING)

TITLE = "Daily News Summary"
ICON = "https://archive.org/favicon.ico"
VISEXP = "https://storage.googleapis.com/data.gdeltproject.org/gdeltv3/iatv/visualexplorer"

BGNDT = pd.to_datetime("2022-03-25").date()
ENDDT = (datetime.now() - timedelta(hours=30)).date()

CHANNELS = {
  "": "-- Select --",
  "ESPRESO": "Espreso TV",
  "RUSSIA1": "Russia-1",
  "RUSSIA24": "Russia-24",
  "1TV": "Channel One Russia",
  "NTV": "NTV",
  "BELARUSTV": "Belarus TV",
  "IRINN": "Islamic Republic of Iran News Network"
}

st.set_page_config(page_title=TITLE, page_icon=ICON, layout="centered", initial_sidebar_state="collapsed")
st.title(TITLE)

llm_predictor = LLMPredictor(llm=OpenAI(max_tokens=1024, model_name="text-davinci-003"))


@st.cache_resource(show_spinner=False)
def load_transcript(id, lg):
  lang = "" if lg == "Original" else ".en"
  r = requests.get(f"{VISEXP}/{id}.transcript{lang}.txt")
  r.raise_for_status()
  return r.content


@st.cache_resource(show_spinner=False)
def load_index(ch, dt, lg):
  r = requests.get(f"{VISEXP}/{ch}.{dt}.inventory.json")
  r.raise_for_status()
  shows = r.json()["shows"]
  idx = GPTVectorStoreIndex.from_documents([], service_context=ServiceContext.from_defaults(llm_predictor=llm_predictor))
  msg = f"Loading `{dt[:4]}-{dt[4:6]}-{dt[6:8]}` {lg} transcripts for `{CHANNELS.get(ch, 'selected')}` channel..."
  prog = st.progress(0.0, text=msg)
  for i, tr in enumerate(shows, start=1):
    try:
      idx.insert(Document(load_transcript(tr["id"], lg).decode("utf-8")), llm_predictor=llm_predictor)
    except HTTPError as e:
      pass
    prog.progress(i/len(shows), text=msg)
  prog.empty()
  return idx.as_query_engine()


@st.cache_resource(show_spinner="Extracting top entities...")
def get_top_entities(_idx, ch, dt, lg):
  res = _idx.query("20 most frequent entities with their frequency in these articles as key-value in JSON format")
  kw = json.loads(res.response.strip())
  wc = WordCloud(background_color="white")
  wc.generate_from_frequencies(kw)
  fig, ax = plt.subplots()
  ax.imshow(wc)
  ax.axis("off")
  return fig, pd.DataFrame(kw.items()).rename(columns={0: "Entity", 1: "Frequency"}).sort_values("Frequency", ascending=False)


@st.cache_resource(show_spinner="Constructing news headlines...")
def get_headlines(_idx, ch, dt, lg):
  return _idx.query("Top 10 news headlines with summary in these articles in Markdown format")


qp = st.experimental_get_query_params()
if "date" not in st.session_state and qp.get("date"):
    st.session_state["date"] = datetime.strptime(qp.get("date")[0], "%Y-%m-%d").date()
if "chan" not in st.session_state and qp.get("chan"):
    st.session_state["chan"] = qp.get("chan")[0]
if "lang" not in st.session_state and qp.get("lang"):
    st.session_state["lang"] = qp.get("lang")[0]

cols = st.columns(3)
dt = cols[0].date_input("Date", value=ENDDT, min_value=BGNDT, max_value=ENDDT, key="date").strftime("%Y%m%d")
ch = cols[1].selectbox("Channel", CHANNELS, format_func=lambda x: CHANNELS.get(x, ""), key="chan")
lg = cols[2].selectbox("Language", ["English", "Original"], format_func=lambda x: "English (Translation)" if x == "English" else x, key="lang", disabled=True) # Disabled due to a bug https://github.com/jerryjliu/gpt_index/issues/294

if not ch:
  st.info(f"Select a channel to summarize for the selected day.")
  st.stop()

st.experimental_set_query_params(**st.session_state)

try:
  idx = load_index(ch, dt, lg)
except HTTPError as e:
  st.warning(f"Transcripts for `{CHANNELS.get(ch, 'selected')}` channel are not available for `{dt[:4]}-{dt[4:6]}-{dt[6:8]}` yet, try selecting another date!", icon="⚠️")
  st.stop()

tbs = st.tabs(["Top Entities", "Frequencies"])
try:
  fig, d = get_top_entities(idx, ch, dt, lg)
  tbs[0].pyplot(fig)
  tbs[1].dataframe(d, use_container_width=True)
except:
  msg = "Entity frequency data is not in the expected JSON shape!"
  tbs[0].warning(msg)
  tbs[1].warning(msg)


"### Top Headlines"

res = get_headlines(idx, ch, dt, lg)
st.markdown(res.response)
