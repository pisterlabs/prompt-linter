import ffmpeg
import requests
import os
import openai
import pytube
from pytube import YouTube
import subprocess

def download_video_mp4(youtube_url):
  # Create a YouTube object
  yt = YouTube(youtube_url)
  
  # Get the video with the highest resolution and file size
  video = yt.streams.filter(progressive=True, 
                            file_extension='mp4').order_by('resolution').desc().first()
  # Download the video to the current working directory
  video.download()
  
  print('Video downloaded!')



  
def create_audio_file(video_filename):
  # Use ffmpeg to extract the audio track from the video and create an .mp4 audio file
  command = 'ffmpeg -i  How Old Is The Water You Drink.mp4 -ab 160k -ar 44100 -vn audio.mp3'
  subprocess.call(command, shell=True)

  

