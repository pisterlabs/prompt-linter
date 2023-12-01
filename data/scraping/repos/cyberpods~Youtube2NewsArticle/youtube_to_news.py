import streamlit as st
import whisper
import openai
from pytube import YouTube
import os
import sys
from pathlib import Path
from zipfile import ZipFile
from gtts import gTTS
import re
import shutil
import tempfile

openai.api_key = os.getenv('OPENAI_API_KEY')

@st.cache_data
def load_model():
    # Load the whisper model
    model = whisper.load_model("base")
    return model

def save_audio(url):
    # Create a temporary directory
    temp_directory = tempfile.mkdtemp()

    # Download the audio file from the given YouTube URL
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download(output_path=temp_directory)
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    try:
        os.rename(out_file, file_name)
    except WindowsError:
        os.remove(file_name)
        os.rename(out_file, file_name)
    audio_filename = os.path.join(temp_directory, Path(file_name).stem+'.mp3')
    st.info(yt.title + " has been successfully downloaded")
    st.audio(audio_filename)
    return yt.title, audio_filename, temp_directory

def limit_string_length(string, max_length):
    # Limit the length of a string to a maximum length
    if len(string) > max_length:
        string = string[:max_length]  # Slice the string up to the maximum length
    return string

def audio_to_transcript(audio_file):
    # Convert audio file to transcript using the loaded whisper model
    model = load_model()
    result = model.transcribe(audio_file)
    transcript = result["text"]
    return transcript

def text_to_news_article(prompt, transcript):
    max_tokens = 4096  # Maximum token limit for API call
    chunk_size = 4000  # Maximum chunk size for each API call
    completion_tokens = 3000  # Maximum tokens for each completion

    # Split the transcript into smaller chunks
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]

    generated_article = ""

    for chunk in chunks:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt + "\n\n" + chunk,
                temperature=0.7,
                max_tokens=completion_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            generated_article += response.choices[0].text
        except openai.OpenAIError as e:
            st.error(f"OpenAI API Error: {str(e)}")
            st.stop()

    return generated_article



def generate_tts_audio(text, filename):
    # Generate an MP3 audio file using gTTS with a female voice
    tts = gTTS(text, lang='en-us')
    tts.save(filename)

def sanitize_file_title(title):
    # Remove invalid characters from the file title
    sanitized_title = re.sub(r'[<>:"/\\|?*]', '', title)
    return sanitized_title

st.markdown('# 📝 **Youtube to News Article Generator App**')

st.header('Input the Video URL')

url_link = st.text_input('Enter URL of YouTube video:')

create_audio = st.checkbox("Create TTS Audio?")

edit_prompt = st.checkbox("Edit Prompt?")

if edit_prompt:
    prompt = st.text_area("Edit Prompt", value="Compose an engaging news article, approximately 500 words long, highlighting the fascinating details surrounding the following text:")
else:
    prompt = "Compose an engaging news article, approximately 500 words long, highlighting the fascinating details surrounding the following text:"

if st.button('Start'):
    with st.spinner('Downloading and processing the audio...'):
        # Download and process the audio file
        video_title, audio_filename, temp_directory = save_audio(url_link)

    with st.spinner('Transcript is being generated...'):
        # Generate the transcript from the audio file
        transcript = audio_to_transcript(audio_filename)
        st.header("Transcript has been generated!")
        st.success(transcript)

    with st.spinner('Generating the news article...'):
        # Generate the news article based on the prompt and transcript
        result = text_to_news_article(prompt, transcript)
        st.header("News Article has been generated!")
        st.success(result)

    if create_audio:
        with st.spinner('Creating TTS audio...'):
            st.markdown("**TTS News Article audio**")
            tts_filename = "tts_news_article.mp3"
            generate_tts_audio(result, tts_filename)
            st.audio(tts_filename, format='audio/mp3')

    # Prepare filenames with the sanitized title
    sanitized_title = sanitize_file_title(video_title)
    title_words = sanitized_title.split()[:5]
    file_title = '_'.join(title_words)

    # Save the transcript and article as text files
    transcript_filename = f"output/transcript_{file_title}.txt"
    transcript_txt = open(transcript_filename, 'w', encoding=sys.getfilesystemencoding())
    transcript_txt.write(transcript)
    transcript_txt.close()

    article_filename = f"output/article_{file_title}.txt"
    article_txt = open(article_filename, 'w', encoding=sys.getfilesystemencoding())
    article_txt.write(result)
    article_txt.close()

    # Create a ZIP file containing the transcript, article, and audio file
    zip_filename = f"output/{file_title}.zip"
    with ZipFile(zip_filename, 'w') as zip_file:
        zip_file.write(transcript_filename)
        zip_file.write(article_filename)
        if create_audio:
            zip_file.write(tts_filename)

    with st.spinner('Saving the files and creating the ZIP...'):
        # Provide download link for the ZIP file
        with open(zip_filename, "rb") as zip_download:
            btn = st.download_button(
                label="Download ZIP",
                data=zip_download,
                file_name=os.path.basename(zip_filename),
                mime="application/zip"
            )

    # Remove the temporary directory
    shutil.rmtree(temp_directory)
