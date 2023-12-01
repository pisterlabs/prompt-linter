import streamlit as st
from langchain.llms import VertexAI
from helpers.vidhelper import streamlit_hide, initialize_llm
from helpers.campaignhelper import build_campaign_from_url
import logging
from dotenv import load_dotenv
import os

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# setup streamlit page
st.set_page_config(page_title="URL Campaign Builder", page_icon="")

streamlit_hide()
st.title("URL Campaign Builder")
st.markdown(
    """ 
                > :black[**Turns a blog into a campaign**]
                """
)
llm, embedding = initialize_llm()

URL = st.text_input(
    "Enter URL from which you need to build a campaign:", placeholder=""
)
submit_button = st.button(label="Build Campaign")

if str(URL) == "":
    st.warning("Awaiting user inputs.")

if submit_button:
    if str(URL) != "":
        with st.spinner("Building campaign..."):
            campaign, msg = build_campaign_from_url(URL)
            if campaign:
                st.success(campaign)
            else:
                st.error(msg)
