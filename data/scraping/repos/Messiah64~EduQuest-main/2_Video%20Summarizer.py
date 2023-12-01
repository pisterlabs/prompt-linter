import openai
from streamlit_pills import pills
import urllib.parse as urlparse
import streamlit as st 
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

OpenAI_Key = st.secrets["OpenAI_Key"]

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

@st.cache_data
def get_keywords_description(keywords):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="provide me a small description in markdown for each of the following " + keywords,
    temperature=0.7,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    
    description = response["choices"][0]["text"]
    return description


@st.cache_data
def get_keywords(transcript):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Extract important keywords mentioned in the following transcript: " + transcript,
    temperature=0.7,
    max_tokens=200,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    keywords = response["choices"][0]["text"]
    return keywords


@st.cache_data
def generateImage(imagePrompt):
    try:
        response = openai.Image.create(
        prompt= imagePrompt,
        n=1,
        size="256x256"
        )

        image_url = response['data'][0]['url']

        return image_url

    except:
        return ""

@st.cache_data
def generatesummary(transcript):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Write a summary in markdown format for the following transcript: " + transcript,
    temperature=0.7,
    max_tokens=200,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    sum = response["choices"][0]["text"]
    return sum

@st.cache_data
def get_transcript(video_id):
    transcript = ""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        for segment in transcript_list:
            transcript += segment['text'] + " "
    except:
        transcript = "Transcript not available for this video."
    return transcript

import urllib.parse as urlparse

def get_videos():
    yt_api_key = st.secrets["Youtube_Key"]
    youtube = build("youtube", "v3", developerKey=yt_api_key)

    video_url = st.text_input('Type in YouTube Video URL')

    url_data = urlparse.urlparse(video_url)
    query = urlparse.parse_qs(url_data.query)
    
    if "v" not in query:
        return []  

    video_id = query["v"][0]
    request = youtube.videos().list(
        part="snippet",
        id=video_id
    )

    response = request.execute()
    videos = []
    for item in response["items"]:
        video_id = item["id"]
        video_title = item["snippet"]["title"]
        videos.append({"title": video_title, "id": video_id})

    return videos


def main():

    st.title("Video Summarizer")
    videos = get_videos()

    for video in videos:

        transcript = get_transcript(video["id"])
        if transcript ==  "Transcript not available for this video.":
            continue

        with st.expander(video["title"], expanded=False):
            tab1, tab2 = st.tabs(["Summary", "Video"])
            transcript = get_transcript(video["id"])
            with tab1:
                col1, col2 = st.columns(2)

                with col1:
                    imagePrompt = video["title"].split("|")[0].strip()
                    image_url = generateImage(imagePrompt)
                    if image_url != "":
                        st.image(image_url)
                    else:
                        st.write("Image cannot be generated!")

                with col2:
                    keywords = get_keywords(transcript)
                    keyword_list = []
                    if keywords.startswith("Keywords: "):

                        keyword_list = keywords[10:].split(", ")
                    else:
                        keyword_list = keywords.split(", ")

                    selected = pills("keywords", keyword_list)

                st.write(generatesummary(transcript))

            with tab2:
                url = "https://www.youtube.com/watch?v=" + video["id"]
                st.video(url)
                st.header("Bytes")
                st.write(get_keywords_description(keywords))

if __name__ == '__main__':
    main()
