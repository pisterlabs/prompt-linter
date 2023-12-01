import logging
import os
from google.cloud import storage
import datetime
import ffmpeg
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
import re
from langchain import PromptTemplate

from langchain.llms import VertexAI

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def streamlit_hide():
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                header {visibility: hidden;}
                footer:before {content : 'Developed by Ankur Wahi';
                        color: tomato;
                        padding: 5px;
                        top: 3px;
                        }
                footer {background-color: transparent;}
                """

    st.markdown(hide_st_style, unsafe_allow_html=True)


def clear_submit():
    st.session_state["submit"] = False


def get_text():
    input_text = st.text_area("Ask a question:", on_change=clear_submit)
    return input_text


@st.cache_data(ttl=180)
def get_video_transcript(video_url, bucket_name):
    try:
        video_id = re.findall(r"v=([^&]+)", video_url)[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        transcript_text = ""
        for segment in transcript:
            text = segment["text"]
            transcript_text += text + " "

        prefix = name_cleanser(video_url)
        transcript_file = prefix + ".txt"

        with open(transcript_file, "w") as file:
            file.write(transcript_text)

        if transcript_file and transcript_text:
            t_file_suc = upload_to_google_storage(bucket_name, transcript_file)

            if t_file_suc:
                logger.info(f"Transcript {transcript_file} has been uploaded")
                return transcript_file, None

        else:
            st.error(
                f"The greatest teacher, failure is...\n. Transcript text not created:{str(transcript_file)}"
            )
            return None, "Transcription failed!"
    except Exception as e:
        logger.error(f"Failed transcription: {str(e)}")
        return None, str(e)


@st.cache_data(ttl=180)
def name_cleanser(name):
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if re.match(r"https?://(?:www\.)?youtube\.com/watch\?v=(.+)", name):
        audio_prefix = name[-6:] + "_" + current_timestamp
        return audio_prefix
    else:
        return None


@st.cache_data(ttl=180)
def upload_to_google_storage(bucket_name, destination_blob_name):
    """
    Uploads a local file to a Google Cloud Storage bucket.
    """

    # Upload the converted audio file to Google Cloud Storage
    try:
        logger.info(
            f"Uploading {destination_blob_name} to Google Cloud Storage bucket: {bucket_name}"
        )
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(destination_blob_name)

        # Delete the local audio files

        os.remove(destination_blob_name)
        return True
    except Exception as e:
        logger.error(
            f"Failed to upload {destination_blob_name} to Google Cloud Storage: {str(e)}"
        )
        return False


def initialize_llm(type="text-bison@001", max_output_tokens=256, temperature=0.1):
    llm = VertexAI(
        model_name=type,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=0.8,
        top_k=40,
        verbose=True,
    )
    embedding = None
    return llm, embedding


@st.cache_data(ttl=180)
def read_file_from_gcs(bucket_name, file_name):
    destination_path = os.getcwd() + "/" + file_name
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_path)
    return destination_path


def langchain_summarize(bucket_name, transcript_file):
    t_file = read_file_from_gcs(bucket_name, transcript_file)

    llm, embedding = initialize_llm()
    logger.info(f"Loading file {str(t_file)} ans using LLM {str(llm)}")
    try:
        loader = TextLoader(t_file)
        documents = loader.load()

        logger.info(f"# of words in the document = {len(documents[0].page_content)}")
    except Exception as e:
        logger.error(f"Unable to load and read transcript {str(t_file)}: {str(e)}")
        for f in os.listdir("."):
            print(f)
        return None, e

    # Get your splitter ready
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)

    # Split your docs into texts
    texts = text_splitter.split_documents(documents)
    combine_prompt = """
    Write a concise summary of the following text delimited by triple backquotes.
    Return your response in bullet points which covers the key points of the text.
    ```{text}```
    BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=["text"]
    )

    try:
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            combine_prompt=combine_prompt_template,
            verbose=False,
        )
        # os.remove(t_file)
        return chain.run(texts), None
    except Exception as e:
        logger.error(f"Langchain summarization failed:{str(e)}")
        return None, e
