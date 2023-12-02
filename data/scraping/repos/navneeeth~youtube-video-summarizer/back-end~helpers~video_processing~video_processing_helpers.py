import threading
import os
import openai
import moviepy.editor as mp
from pytube import YouTube
from models.id.id_model import ID
from helpers.id.id_helpers import store_id
from config import db

MAX_THREADS = 5
active_threads = []

def download_video(audio_file, video_file, video_link):
    youtube = YouTube(video_link, use_oauth=True, allow_oauth_cache=True)
    audio = youtube.streams.filter(only_audio=True).first()
    audio.download(filename=video_file)
    mp.AudioFileClip(video_file).write_audiofile(audio_file)

def transcribe_audio(file):
    return openai.Audio.transcribe("whisper-1", file)["text"]

def generate_summary(transcription_text):
    prompt = "Organize this transcription from a YouTube video into a structured set of easily understandable points without missing important details: " + transcription_text
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}]
    )
    return summary.choices[0].message.content if len(summary.choices) > 0 else ""

def update_processing_status(acknowledgement_id, timestamp, video_link, video_title, summary, status):
    try:
        status_update = ID(
            acknowledgement_id=acknowledgement_id,
            timestamp=timestamp,
            video_link=video_link,
            video_title=video_title,
            summary=summary,
            status=status
        )
        store_id(status_update.__dict__)
    except Exception as e:
        print("Error:", e)

def remove_any_mp3_files():
    """
    Remove all .mp3 files in the current directory.
    """
    # Get the current directory
    current_directory = os.getcwd()

    # List all files in the current directory
    file_list = os.listdir(current_directory)

    # Iterate through the files and delete .mp3 files
    for file in file_list:
        if file.endswith(".mp3"):
            file_path = os.path.join(current_directory, file)
            os.remove(file_path)

def process_request(new_id, timestamp, video_link, video_title):
    print('Started thread')
    print(threading.current_thread().name)
    try:
        remove_any_mp3_files()
        
        # Update status to 'Downloading'
        update_processing_status(new_id, timestamp, video_link, video_title, '', 'Downloading')
        
        audio_file = video_title + ".mp3"
        video_file = video_title + ".mp4"

        # Download and convert video
        download_video(audio_file, video_file, video_link)

        # Update status to 'Transcribing'
        update_processing_status(new_id, timestamp, video_link, video_title, '', 'Transcribing')
        
        # Transcribe audio
        file = open(audio_file, "rb")
        transcription_text = transcribe_audio(file)
        with open(video_title+"_transcription.txt", "w", encoding="utf-8") as f:
            f.write(transcription_text)
        
        # Update status to 'Summarizing'
        update_processing_status(new_id, timestamp, video_link, video_title, '', 'Summarizing')
        
        # Generate summary
        summary_text = generate_summary(transcription_text)
        # print the summary and write it to a text file
        with open(video_title+"_summary.txt", "w") as f:
            f.write(summary_text)
        # Update status to 'Ready' and store summary
        update_processing_status(new_id, timestamp, video_link, video_title, summary_text, 'Ready')
        
        # Remove temporary files
        os.remove(video_title+"_summary.txt")
        os.remove(video_title+"_transcription.txt")
        os.remove(video_file)
        #os.remove(audio_file)
    
    except Exception as e:
        print("Error:", e)
        # Update status to 'Error'
        update_processing_status(new_id, timestamp, video_link, video_title, '', 'Error')
        return
    #active_threads.stop()
    
def process_request_legacy(new_id, timestamp, video_link, video_title):
    print('Started thread')
    print(threading.current_thread().name)
    db.collection('auth').document(new_id).set({'timestamp': timestamp, 'status': 'Downloading', 'video_link': video_link, 'video_title': video_title, 'summary': ''})
    video_file = video_title + ".mp4"
    audio_file = video_title + ".mp3"
    updated_audio_file = "updated_" + audio_file
    youtube = YouTube(video_link, use_oauth=True, allow_oauth_cache=True)
    print(youtube)
    audio = youtube.streams.filter(only_audio=True).first()
    audio.download(filename=video_file)
    # convert the downloaded audio file to mp3 format
    mp.AudioFileClip(video_file).write_audiofile(audio_file)
    print("Processing finished for timestamp:", timestamp, "and video link:", video_link)
    db.collection('auth').document(new_id).set({'timestamp': timestamp, 'status': 'Transcribing', 'video_link': video_link, 'video_title': video_title, 'summary': ''})
    # transcribe the audio using OpenAI's API
    file = open(audio_file, "rb")
    transcription = openai.Audio.transcribe("whisper-1", file)

    # write the transcription to a text file
    with open(video_title+"_transcription.txt", "w", encoding="utf-8") as f:
        f.write(transcription["text"])
    db.collection('auth').document(new_id).set({'timestamp': timestamp, 'status': 'Summarizing', 'video_link': video_link, 'video_title': video_title, 'summary': ''})
    prompt = "Organize this transcription from a YouTube video into a structured set of easily understandable points without missing important details: "+transcription["text"]
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}]
    )
    text = ""
    print(summary.choices)
    if len(summary.choices) > 0:
        text = summary.choices[0].message.content
        print(text)
    else:
        print("Error: No response generated.")
        db.collection('auth').document(new_id).set({'timestamp': timestamp, 'status': 'Error', 'video_link': video_link, 'video_title': video_title, 'summary': ''})
    # print the summary and write it to a text file
    with open(video_title+"_summary.txt", "w") as f:
        f.write(text)
    db.collection('auth').document(new_id).set({'timestamp': timestamp, 'status': 'Ready', 'video_link': video_link, 'video_title': video_title, 'summary': text})
    os.remove(video_title+"_summary.txt")
    os.remove(video_title+"_transcription.txt")
    os.remove(video_file)
    os.remove(audio_file)
    #active_threads.stop()
    
    

def start_processing_thread(new_id, timestamp, video_link, video_title):
    global active_threads
    # Check if there are already MAX_THREADS active threads
    if len(active_threads) >= MAX_THREADS:
        # Wait for one of the threads to complete
        active_threads[0].join()
        # Remove the completed thread from the list
        active_threads = active_threads[1:]
    # Create a new thread for the request
    t = threading.Thread(target=process_request, args=(new_id, timestamp, video_link, video_title))
    # Add the thread to the list of active threads
    active_threads.append(t)
    # Start the thread
    t.start()
