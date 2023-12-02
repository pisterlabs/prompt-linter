"""A basic chatbot using the OpenAI API + Community Notion Info"""
import logging
import os
import sys
from typing import Any, Dict, Generator, List, Union
from urllib.parse import parse_qs, urlparse

import anthropic
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

ResponseType = Union[Generator[Any, None, None], Any, List, Dict]

# Load the .env file
load_dotenv()

# Set up the OpenAI API key
assert os.getenv("ANTHROPIC_API_KEY"), "Please set your ANTHROPIC_API_KEY environment variable."
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


def download(video: str) -> str:
    """Download the video transcript."""
    print(f"Downloading {video}...")
    parsed_url = urlparse(video)
    video_id = parse_qs(parsed_url.query)["v"][0]
    loader = YoutubeLoader(video_id)
    docs = loader.load()
    tx = str(docs[0].page_content)
    return tx


def main() -> None:
    """Run the chatbot."""
    st.title("Ask Milo about MLOps.Community Videos👋")
    if not anthropic_api_key:
        st.error("Please set your ANTHROPIC_API_KEY environment variable.")
        st.stop()

    youtube_article = st.text_input(
        "Enter a YouTube URL",
        placeholder="https://www.youtube.com/watch?v=0e5q4zCBtBs",
    )
    if not youtube_article:
        st.error("Please enter a YouTube URL.")
        st.stop()
    if "v=" not in youtube_article:
        st.error("Please enter a YouTube URL with video id e.g. `v=`.")
        st.stop()

    with st.spinner("Downloading video transcription..."):
        transcript = download(youtube_article)
        if not transcript:
            st.error("No transcript found.")
            st.stop()

    question = st.text_input(
        "Ask something about the article",
        placeholder="Can you give me a short summary?",
    )
    if question:
        with st.spinner("Thinking..."):
            prompt = f"""{anthropic.HUMAN_PROMPT} Here's the transcript of a MLOps Commuinity video:
    {transcript}

    Answer the question below using only the information in the transcription.
    Question: {question}

    {anthropic.AI_PROMPT}"""

            client = anthropic.Client(api_key=anthropic_api_key)
            response = client.completions.create(
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-2",  # "claude-2" for Claude 2 model
                max_tokens_to_sample=256,
            )
            st.write("### Answer")
            st.write(response.completion)


if __name__ == "__main__":
    main()
