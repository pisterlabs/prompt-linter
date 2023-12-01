"""Streamlit app for the Generative AI prototypes."""

import os

import openai
import streamlit as st

from dotenv import load_dotenv

from genai.streamlit_pages import parenting_chatbot


load_dotenv()


def auth_openai() -> None:
    """Authenticate with OpenAI."""
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except Exception:
        openai.api_key = st.secrets["OPENAI_API_KEY"]


def s3_creds() -> None:
    """Get s3 creds."""
    try:
        aws_key = os.environ["AWS_ACCESS_KEY_ID"]
        aws_secret = os.environ["AWS_SECRET_ACCESS_KEY"]
        s3_path = os.environ["S3_BUCKET"]
    except Exception:
        aws_key = st.secrets["AWS_ACCESS_KEY_ID"]
        aws_secret = st.secrets["AWS_SECRET_ACCESS_KEY"]
        s3_path = st.secrets["S3_BUCKET"]

    return aws_key, aws_secret, s3_path


def main() -> None:
    """Run the app."""
    auth_openai()
    aws_key, aws_secret, s3_path = s3_creds()
    parenting_chatbot(aws_key, aws_secret, s3_path, sidebar=False)


main()
