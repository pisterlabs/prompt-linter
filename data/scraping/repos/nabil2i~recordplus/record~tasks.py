from __future__ import absolute_import, unicode_literals

from celery import shared_task
from time import sleep
import openai
from .models import RecordedVideo, Transcription
from openai.error import OpenAIError
# import whisper

@shared_task
def transcribe_video(video_id):
  # sleep(10)
  print("Transcribing video...")
  try:
    # get video path
    print(f"video id: {video_id}")
    video = RecordedVideo.objects.get(pk=video_id)
    video_file_path = video.get_video_file_url()
    
    if not video_file_path:
      print(f"Video with ID {video_id} not found")
      return
      
    with open(video_file_path, 'rb') as video_file:
      response = openai.Audio.transcribe("whisper-1", video_file)
      
      # model = whisper.load_model("base")
      # response = model.transcribe(video_file)
    # with open(video_file_path, 'rb') as video_file:
    #   response = openai.Transcription.create(
    #     audio=video_file,
    #     engine="whisper",
    #     language="en-US",
    #     max_tokens=300,
    #   ) 
    
    if 'text' in response:
      transcription_text = response['text']
      print(transcription_text)
    
      transcription, created = Transcription.objects.get_or_create(
        recorded_video=video,
        defaults={'transcription_text': transcription_text}
      )
      
      if not created:
        transcription.transcription_text = transcription_text
        transcription.save()
    
      print("Video transcribed")
    
    else:
      print(f"Error in OpenAI response: {response}")
  
  except RecordedVideo.DoesNotExist:
    print(f"Video with ID {video_id} does not exist.")
  except OpenAIError as e:
    print(f"OpenAI Error: {str(e)}")
  except Exception as e:
    print(f"Error transcribing the video {video_id}: {str(e)}")
