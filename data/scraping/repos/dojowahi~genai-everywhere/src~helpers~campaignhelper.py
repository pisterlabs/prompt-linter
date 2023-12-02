import pandas as pd
import json
import time

import datetime
import requests
import logging
import os
import streamlit as st
from google.cloud import bigquery
import os

# from langchain import OpenAI
from helpers.vidhelper import initialize_llm, upload_to_google_storage
from vertexai.preview.language_models import (
    TextGenerationModel,
)
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_campaign_from_url(URL):
    try:
        generation_model = TextGenerationModel.from_pretrained("text-bison@001")
        prompt = f""" Summarize this {URL} and create an ad copy for a Google Ads campaign """

        response = generation_model.predict(prompt=prompt)

        response = generation_model.predict(
            prompt=prompt, max_output_tokens=1024, temperature=0.8, top_k=30, top_p=0.7
        )
        return response.text, None
    except Exception as e:
        logger.error(f"Error {str(e)}")

        return None, str(e)
