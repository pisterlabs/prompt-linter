import uuid
import time
import os
import boto3
import requests
from pydub import AudioSegment
from yt_dlp import YoutubeDL
from flask import Flask, request, jsonify
from flask_cors import CORS
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# Audio Processing Functions
def download_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio',
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def upload_to_s3(filename, bucket_name):
    s3 = boto3.client('s3')
    s3.upload_file(filename, bucket_name, filename)
    return f"s3://{bucket_name}/{filename}"

def transcribe_audio(job_name, job_uri):
    transcribe = boto3.client('transcribe', region_name='us-west-1')
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        LanguageCode='en-US',
        Settings={'ShowSpeakerLabels': True, 'MaxSpeakerLabels': 10}
    )
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(15)
    return status['TranscriptionJob']['Transcript']['TranscriptFileUri']

def get_text_from_transcription(transcript_uri):
    response = requests.get(transcript_uri)
    transcript = response.json()
    items = transcript['results']['items']
    turns = []
    current_speaker = None
    sentence = ""
    for item in items:
        if 'type' in item and item['type'] == 'pronunciation':
            if 'speaker_label' in item and current_speaker != item['speaker_label'] and sentence:
                turns.append({'speaker': current_speaker, 'content': sentence.strip()})
                sentence = ""
            current_speaker = item['speaker_label']
            sentence += " " + item['alternatives'][0]['content']
        elif 'type' in item and item['type'] == 'punctuation':
            sentence += item['alternatives'][0]['content']
    if sentence:
        turns.append({'speaker': current_speaker, 'content': sentence.strip()})
    return turns

def format_transcript(turns):
    formatted_transcript = ""
    for turn in turns:
        formatted_transcript += f"{turn['speaker']}: {turn['content']}\n"
    return formatted_transcript

def parse_audio(youtube_url, bucket_name="transcribe-youtube-distinct-speakers"):
    job_name = f"TranscribeJob-{uuid.uuid4()}"
    download_audio(youtube_url)
    job_uri = upload_to_s3("audio.wav", bucket_name)
    transcript_uri = transcribe_audio(job_name, job_uri)
    text = get_text_from_transcription(transcript_uri)
    return text

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
anthropic = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
prompt = ""

# API Endpoints
@app.route('/parse_audio', methods=['POST'])
def parse_audio_endpoint():
    global prompt
    youtube_url = request.json['youtube_url']
    text = parse_audio(youtube_url)
    formatted_text = format_transcript(text)
    prompt = f"{HUMAN_PROMPT} {formatted_text}"
    return jsonify({"message": "Audio parsed and prompt updated", "text": formatted_text})

@app.route('/chat', methods=['POST'])
def chat():
    global prompt
    user_input = request.json['user_input']
    prompt += f"{HUMAN_PROMPT} {user_input}{AI_PROMPT}"
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=10000,
        prompt=prompt
    )
    output = completion.completion
    prompt += output
    return jsonify({"message": "Chat updated", "output": output})

# Main Function
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
