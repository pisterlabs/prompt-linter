"""Streamlit app for the Generative AI prototypes."""

import os

import openai
import streamlit as st

from dotenv import load_dotenv

from genai.streamlit_pages import eli3


load_dotenv()


def auth_openai() -> None:
    """Authenticate with OpenAI."""
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except Exception:
        openai.api_key = st.secrets["OPENAI_API_KEY"]


def main() -> None:
    """Run the app."""
    auth_openai()
    eli3(sidebar=False)


main()
