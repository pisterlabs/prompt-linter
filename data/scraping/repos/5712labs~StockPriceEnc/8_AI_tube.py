import streamlit as st
from components import convert
from pytube import YouTube
from moviepy.editor import *
import openai
import os
from datetime import datetime

title = f'AI DWTUBE {__name__}'
st.set_page_config(page_title=title, page_icon="🐍", layout="wide")
st.title(title)

if convert.check_password() == False:
    st.stop()

st.sidebar.write('https://www.youtube.com/watch?v=6Af6b_wyiwI')

url_input = st.text_input(
    "Enter the URL of the YouTube video: 👇",
    # label_visibility=st.session_state.visibility,
    # disabled=st.session_state.disabled,
    # placeholder="Enter the URL of the YouTube video: ",
)

if url_input:
  # st.write("You entered: ", url_input)
  url = url_input
  try:
    # Download the YouTube video
    video = YouTube(url)
    # st.video(video, format="video/mp4")
    # url = input('Enter the URL of the YouTube video: ')
    # prompt = 'https://www.youtube.com/watch?v=reUZRyXxUs4'
    # prompt = 'https://www.youtube.com/watch?v=EZmiYr9Q2N4'
    # prompt = 'https://www.youtube.com/watch?v=6Af6b_wyiwI'
    
    st.video(url)
    st.info(f'{video.title} by {video.author}')
    # st.write(video.length)
    # st.write(video.publish_date)
    # st.write(video.views)
    # st.write(video.keywords)
    # st.write(video.description)
    # st.write(video.thumbnail_url)

    audio_stream = video.streams.filter(only_audio=True).first()
    audio_stream.download(output_path='./youtube/tmp/')
    # Convert the audio file to MP3
    input_file = './youtube/tmp/' + audio_stream.default_filename
    audio_clip = AudioFileClip(input_file)
    # today = datetime.today().strftime('%Y년 %m월 %d일 %H시%M분')
    today = datetime.today().strftime('%Y%m%d%H%M_')

    output_file = './youtube/' + today + audio_stream.title + '.mp3'
    audio_clip.write_audiofile(output_file)
    st.audio(output_file)
    st.write(video.publish_date)
    st.write(audio_stream.title)

    # Remove the original audio file
    os.remove(input_file)
    # os.remove(output_file)

    file = open(output_file, "rb")
    # Transcribe the audio file
    transcription = openai.Audio.transcribe("whisper-1", file)

    # Write the transcription to a file
    with open(output_file, "w") as file:
        file.write(transcription['text'])

    with open(output_file, "r") as file:
        content = file.read()
        st.write(content)
    
  except:
    st.error('영상을 불러올 수 없습니다.')
    st.stop()

st.stop()  
if prompt := st.chat_input("YouTube 주소를 넣어주세요"):
  
  st.video(url)
  # Download the YouTube video
  video = YouTube(url)
  st.write(video.title)
  st.write(video.length)
  st.write(video.author)
  st.write(video.publish_date)
  st.write(video.views)
  st.write(video.keywords)
  st.write(video.description)
  st.write(video.thumbnail_url)

  # st.video(video, format="video/mp4")
  audio_stream = video.streams.filter(only_audio=True).first()
  audio_stream.download(output_path='./youtube/')

  # Convert the audio file to MP3
  input_file = './youtube/' + audio_stream.default_filename
  
  st.write(audio_stream)
  st.write(audio_stream.title)
  st.write(audio_stream.default_filename)
  st.write(audio_stream.filesize_mb)
  

  audio_clip = AudioFileClip(input_file)
  audio_clip.write_audiofile(output_file)

  # Remove the original audio file
  os.remove(input_file)


  # Open the audio file
  file = open("./youtube/audio.mp3", "rb")
  st.stop()

  # Transcribe the audio file
  transcription = openai.Audio.transcribe("whisper-1", file)

  # Write the transcription to a file
  with open(output_file, "w") as file:
      file.write(transcription['text'])















  with open(scriptpath, "r") as file:
      content = file.read()
      st.write(content)
      # print(f"The extracted script is : {content}")

st.stop()









def extract_audio(url, output_file):
   
    st.video(url)
    # Download the YouTube video
    video = YouTube(url)
    st.write(video.title)
    st.write(video.length)
    st.write(video.author)
    st.write(video.publish_date)
    st.write(video.views)
    st.write(video.keywords)
    st.write(video.description)
    st.write(video.thumbnail_url)

    # st.video(video, format="video/mp4")
    audio_stream = video.streams.filter(only_audio=True).first()
    audio_stream.download(output_path='./youtube/')

    # Convert the audio file to MP3
    input_file = './youtube/' + audio_stream.default_filename
    
    st.write(audio_stream)
    st.write(audio_stream.title)
    st.write(audio_stream.default_filename)
    st.write(audio_stream.filesize_mb)
    

    audio_clip = AudioFileClip(input_file)
    audio_clip.write_audiofile(output_file)

    # Remove the original audio file
    os.remove(input_file)


def youtube_to_script(url, scriptpath):
    # Extract the audio from the YouTube video
    extract_audio(url, './youtube/audio.mp3')

    # Open the audio file
    file = open("./youtube/audio.mp3", "rb")
    st.stop()

    # Transcribe the audio file
    transcription = openai.Audio.transcribe("whisper-1", file)

    # Write the transcription to a file
    with open(output_file, "w") as file:
        file.write(transcription['text'])


def main():
    if prompt := st.chat_input("YouTube 주소를 넣어주세요"):
      # url = input('Enter the URL of the YouTube video: ')
      prompt = 'https://www.youtube.com/watch?v=EZmiYr9Q2N4'
      prompt = 'https://www.youtube.com/watch?v=6Af6b_wyiwI'
      youtube_to_script(prompt, scriptpath)

      with open(scriptpath, "r") as file:
          content = file.read()
          st.write(content)
          # print(f"The extracted script is : {content}")

if __name__ == '__main__':
    main()