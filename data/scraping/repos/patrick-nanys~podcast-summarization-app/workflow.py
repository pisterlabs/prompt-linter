import subprocess
import csv
import json
import os
import io
import openai
import boto3

import pandas as pd
import numpy as np
from pytube import YouTube
from elevenlabslib import ElevenLabsUser

from experiments.asr.asr import whisper_asr
from experiments.tts.tts import elevenlabs_tts
from experiments.summarization.openai.summarize import summarize_text, num_tokens_from_text


def download_video(video_url, destination):
    yt = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
    print(f"Downloading video: {yt.title} from author {yt.author}")
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download(output_path=destination)
    base, ext = os.path.splitext(out_file)
    new_target_file_name = base + ".wav"
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            out_file,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            "96K",
            "-acodec",
            "pcm_s16le",
            new_target_file_name,
        ]
    )

    return new_target_file_name


def load_files_and_timestamps(target_dir, target_file_name, output_directory):
    """
    Reading back the text files. This could be done later asynchronously/with multithreading
    Also saves the transcription df to @param output_directory
    """
    target_file_name_base, ext = os.path.splitext(target_file_name)
    dfs = []

    # This trick is needed to sort the filenames by index instead of alphabetically
    correct_files = []
    for file_name in os.listdir(target_dir):
        if target_file_name_base in file_name and file_name.endswith("csv"):
            correct_files.append(file_name)

    base_file_name = correct_files[0][: correct_files[0].rfind("_")]
    for file_idx in range(len(correct_files)):
        file_name = base_file_name + f"_{file_idx}" + ".csv"
        print(file_name)
        dfs.append(
            pd.read_csv(
                os.path.join(target_dir, file_name),
                delimiter=";",
                names=["start", "end", "text"],
                encoding="ISO-8859-1",
                quoting=csv.QUOTE_NONE,
            )
        )

    df = pd.concat(dfs).reset_index(drop=True)
    df["text"] = df["text"].astype(str)

    final_lines = " ".join(df["text"])

    df["text_token_counts"] = df["text"].map(num_tokens_from_text)
    df["token_sum"] = np.cumsum(df["text_token_counts"])

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    df.to_csv(os.path.join(output_directory, "transcription.csv"), index=False, sep=";")

    token_sums = [0] + list(df["token_sum"])
    timestamp_values = list(df["end"])
    timestamp_values.insert(0, df["start"].iloc[0])
    timestamps_dict = dict(zip(token_sums, timestamp_values))

    return final_lines, timestamps_dict


def list_files_in_bucket(bucket_name, s3):
    my_bucket = s3.Bucket(bucket_name)
    for my_bucket_object in my_bucket.objects.all():
        print(my_bucket_object.key)


def upload_directory_to_bucket(path, bucket_name, s3):
    for file_name in os.listdir(path):
        s3.meta.client.upload_file(os.path.join(path, file_name), bucket_name, f"podcasts/{path}/{file_name}")


def workflow(video_url, video_download_folder, output_directory, ELEVENLABS_API_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY):
    # Download video

    target_file_name = download_video(video_url, video_download_folder)
    target_file_name = os.path.basename(target_file_name)
    target_file_name_base, ext = os.path.splitext(target_file_name)
    podcast_sound_path = os.path.join(video_download_folder, target_file_name)

    # Speech to text

    whisper_asr(podcast_sound_path)

    # Loading transcriptions, saving it

    text, timestamps_dict = load_files_and_timestamps(video_download_folder, target_file_name, output_directory)

    # Text summary, saving the generated files to json

    input_text, chunks, chunk_start_timestamps = summarize_text(text, timestamps_dict)

    with open(os.path.join(output_directory, "summarized_text.json"), "w") as f:
        json.dump(input_text, f, indent=2)
    with open(os.path.join(output_directory, "chunks.json"), "w") as f:
        json.dump(chunks, f, indent=2)
    with open(os.path.join(output_directory, "chunk_start_timestamps.json"), "w") as f:
        json.dump(chunk_start_timestamps, f, indent=2)

    # Check if ElevenLabs can be used

    summary_length = sum([len(chunk) for chunk in chunks])
    print("Summary length:", summary_length)

    user = ElevenLabsUser(ELEVENLABS_API_KEY)
    remaining_characters = user.get_character_limit() - user.get_current_character_count()
    print("Remaining ElevenLabs characters:", remaining_characters)

    if summary_length > remaining_characters:
        raise ValueError(
            "Not enough characters for TTS. Provide an ElevenLabs API token with enough remaining characters."
        )

    # TTS with elevenlabs

    elevenlabs_tts(
        chunks, ELEVENLABS_API_KEY, "Adam", os.path.join(output_directory, "read_summary.mp3")
    )  # Male voice

    # create config

    config = {
        'name': target_file_name_base
    }
    with open(os.path.join(output_directory, 'config.json'), 'w') as f:
        json.dump(config, f)

    # upload to s3 bucket

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )
    s3 = session.resource("s3")
    bucket_name = "breviocast-prod"

    upload_directory_to_bucket(output_directory, bucket_name, s3)
    list_files_in_bucket(bucket_name, s3)


if __name__ == "__main__":
    # Set parameters

    video_url = ""
    video_download_folder = "downloaded_videos_test"
    output_directory = ""
    ELEVENLABS_API_KEY = ""
    AWS_ACCESS_KEY = ""
    AWS_SECRET_KEY = ""
    openai.api_key = ""

    # Run the script
    workflow(video_url, video_download_folder, output_directory, ELEVENLABS_API_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY)
