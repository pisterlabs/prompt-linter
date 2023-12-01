from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,    
    HumanMessagePromptTemplate,
)
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from pytube import YouTube

OPENAI_API = st.secrets["OPENAI_API"]

template = (""" 
    I'm preparing notes from a video titled "{title}". I need you to
    act as an expert professor and provide me with comprehensive and well-structured 
    notes from the following text. 

    Here is the text:
    {transcription}

    Condition: Please ensure the notes cover the following topics: ALL THE TOPICS.
    Also do make sure you are printing everything in MARKDOWN. 
    Strictly do not print anything else other than the notes.
    """)

def video_title(url):    
    """
    This function retrieves the title of a YouTube video given its URL.

    Arguments:
    url -- A string representing the URL of a YouTube video.

    Returns:
    str -- A string representing the title of the YouTube video.

    Raises:
    Exception -- If the provided URL is not valid or does not point to a YouTube video.
    """
        
    yt = YouTube(url)
    return yt.title

def text_extractor(url):
    """
    This function extracts the text from a YouTube video transcript given its URL.

    Args:
        url: A string representing the URL of a YouTube video.

    Returns:
        A string containing the transcript text of the YouTube video.

    Raises:
        ConnectionError: If there is an error connecting to the YouTube Transcript API.
    """
    try:
        if "&list=" in url:
            url = url.split("&list=")[0]
        elif "?si=" in url:
            url = url.split("?si=")[0]
        video_id = url.split("?v=")[-1] if "?v=" in url else url.split("/")[-1]

    except IndexError:
        video_id = url.split("/")[-1]

    try:
        response = YouTubeTranscriptApi.get_transcript(video_id)
        final = "".join([i['text'] for i in response])

        if 4078 > len(final) > 5:
            return final
        else:
            return None
    except ConnectionError as e:
         st.error(e)

@st.cache_data
def notes_generator(url):
    """
    This function generates notes based on the provided URL.

    Args:
        url: A string representing the URL of the content for which notes are to be generated.

    Returns:
        A string containing the generated notes.

    Raises:
        Exception: If the URL is not valid or if there's an error in generating notes.

    """
    chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API)

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    gpt_response = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    response = chat(
        gpt_response.format_prompt(
            title=video_title(url), transcription=text_extractor(url) if text_extractor(url) is not None else "Sorry, couldn't extract the transcript for this video.", 
            text=url
        ).to_messages()
    )

    return response.content

def credits(url):
     """
    This function generates credits of the content-creator based on the provided URL.

    Args:
        url: A string representing the URL of the content for which notes are to be generated.

    Returns:
        A string containing the credits for the YouTube video.

    Raises:
        Exception: If the URL is not valid or if there's an error in generating notes.

    """
     yt = YouTube(url)
     return yt.title, yt.author, yt.channel_url, yt.publish_date, yt.thumbnail_url