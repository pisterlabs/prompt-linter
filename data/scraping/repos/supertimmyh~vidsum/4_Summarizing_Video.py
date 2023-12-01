import os
import glob
from pydub import AudioSegment
import openai
import yt_dlp
import streamlit as st
import math

# Set the OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.Client()

# Function to extract audio from video
def extract_audio(video_directory):
  video_file = glob.glob(video_directory)[0]
  audio_file = video_file.replace(".mp4", ".wav")
  audio = AudioSegment.from_file(video_file)
  audio.export(audio_file, format="wav")
  # start the segmentation process
  # get the audio file size and OpenAI audio permittd size
  full_audio_size = os.path.getsize(audio_file)
  openai_audio_size = 25000000

  # calcualte the length of OpenAI audio length in the audio file
  segment_duration = audio.duration_seconds * (openai_audio_size/full_audio_size)

  # get the number of segments
  # for an uneven number of segment, keep the last, unfitting part
  # of the audio file and round to next integer number
  number_of_segments = math.ceil(int(audio.duration_seconds/segment_duration))

  # specify the starting prefix number and first segment's begining and ending
  # Create a list to store the exported audio files
  exported_segment_files = []
  prefix = 1
  segment_start = 0
  segment_end = segment_duration*1000
  

  for i in range(number_of_segments+1):
    segment_export = audio[segment_start:segment_end]

    # define the segment file name
    segment_file = audio_file.replace(".wav", str(i) + ".wav")
    # export the segement to new wav file
    segment_export.export(segment_file, format="wav")
    # append the segment file name to the list
    exported_segment_files.append(segment_file)

    # increase the beginning and end of the segment by duration length
    # in order to get to the next segment
    segment_start += segment_duration*1000
    # if the last segment is shorter than the OpenAI audio size
    # then use the remaining audio length
    if (i == (number_of_segments - 1)):
      segment_end = audio.duration_seconds - (segment_duration*(i+1))

    else:
      segment_end += segment_duration*1000

    # return if the next segment is less than 0.1 seconds
    if (segment_end - segment_start) < 100:
      return exported_segment_files

    # iterate the loop count
    i += 1
  
  # clean up the audio file and return the list of segment files
  os.remove(audio_file)
  return exported_segment_files


# Function to transcribe audio using Whisper
def transcribe_audio(exported_segment_files):
  # loop over the list of exported_segment_files
  for segment_file in exported_segment_files:
    # load the model
    with open(segment_file, 'rb') as audio_source:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_source,
            response_format="text"
      )
    os.remove(segment_file)
    transcript =+ transcript

  return transcript

# Function to clean up the transcript using OpenAI
def cleanup_transcript(text):
  response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    temperature=0.1,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.5,
    max_tokens=3000,
    messages=[
        {"role": "system", "content": "You are a university graduate with a masters degree in English"},
        {"role": "user", "content": "I will provide you with a transcription from a video. You will summarize the transcription in the tldr fashion. Extract the most important key points and use them as markdown formatted headings. Give a detailed extractive and abstract summary for each key point. It is important that you are very specific and clear in your response."},
        {"role": "user", "content": text}
    ]
)
  return response["choices"][0]["text"]

# Function to create a summary of the transcript
def summary(text):
  response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    temperature=0.1,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.5,
    max_tokens=3000,
    messages=[
        {"role": "system", "content": "You are a university graduate with a masters degree in English"},
        {"role": "user", "content": "I will provide you with a transcription from a video. You will provide a 1 sentence tldr. Extract the most important key points and use them as markdown formatted headings. Give a detailed extractive and abstract summary for each key point.  It is important that you are very specific and clear in your response. Conclude with a one paragraph abstract summary of what the author wanted to convince us of. \n\nVideo transcript:\n{text}\nSummary:"},
        {"role": "user", "content": text}
    ]
)
  return response["choices"][0]["text"]

def summarizing_video():
  video_url = st.text_input("Enter the URL of the video you want to summarize: ")

  # Download the video
  yt_dlp.YoutubeDL({
    'ignoreerrors': True,
    'writeinfojson': True,
    'addmetadata': True,
    'writesubtitles': True,
    'subtitleslangs': ['en', 'de', 'ja'],
    'writethumbnail': True,
    'embedsubs': True,
    'format': 'mp4',
    'outtmpl': './videos/%(title)s.%(ext)s'
  }).download([video_url])

  # Directory where the video is stored
  video_directory = os.path.join("./videos", "*.mp4")

  # Extract audio from video
  audio_file = extract_audio(video_directory)

  # Transcribe the audio
  raw_transcript = transcribe_audio(audio_file)

  # Clean up the transcript
  legible_transcript = cleanup_transcript(raw_transcript)

  # Create a summary of the transcript
  summary_text = summary(legible_transcript)

  # Display a success message
  st.success('Success! Your summary is below: /n' + summary_text)

# Config the app
st.set_page_config(page_title="Video Summarization", page_icon="📈")
st.markdown("# Video Summarization")
st.sidebar.header("Video Summarization")
st.write(
    """This demo illustrates a combination of video summarization and natural language processing with
Streamlit. We're downloading a video from YouTube and summarizing it. Enjoy!"""
)

summarizing_video()