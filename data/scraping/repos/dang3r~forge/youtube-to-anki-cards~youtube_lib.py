import subprocess
from youtube_transcript_api import YouTubeTranscriptApi
import json
import os

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import google.oauth2.credentials
import openai

from pytube import YouTube

#  Playlist ID for the playlist of videos to summarize
playlist_id = "PL2EVlV9VQTwRMtq4WMSxkn8If61JgUtkt"
scopes = ["https://www.googleapis.com/auth/youtube.readonly"]


def download_audio(video_id, output_directory, output_filename):
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(url)
        audio = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        print(f"Downloading audio: {yt.title}, {output_filename}")
        audio.download(output_directory, filename=output_filename)
        print(f"Downloaded audio: {yt.title}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error downloading audio: {e}")


def whisper(audio_file):
    subprocess.run(
        f"ffmpeg -i {audio_file} -acodec pcm_s16le -ac 1 -ar 16000 {audio_file}.wav".split(),
        check=True,
    )
    subprocess.run(
        f"./whisper --model ggml-base.en.bin --file {audio_file}.wav --output-txt --output-file {audio_file}.wav --no-timestamps".split(),
        check=True,
    )


def summarize(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": f"Please summarize the following transcript and provide a list of 5 questions intended to test the student's understanding of the material.:\n{text}",
            },
        ],
    )
    return response


def youtube_videos():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "client_secret.json"

    # TODO: Refresh auth token periodically
    # Get credentials and create an API client
    # UNCOMMENT THIS TO GET NEW CREDENTIALS
    # flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
    #    client_secrets_file, scopes
    # )
    # credentials = flow.run_console()
    # print(credentials.to_json())
    # print(credentials)
    credentials = google.oauth2.credentials.Credentials.from_authorized_user_file(
        "oauth.json", scopes=scopes
    )
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials
    )

    request = youtube.playlistItems().list(
        part="snippet,contentDetails", maxResults=25, playlistId=playlist_id
    )
    response = request.execute()
    for item in response["items"]:
        title = item["snippet"]["title"]
        video_id = item["contentDetails"]["videoId"]
        yield title, video_id
