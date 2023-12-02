import sys

import openai
from youtube_transcript_api import (YouTubeTranscriptApi, NoTranscriptFound)

from config import OPENAI_API_KEY


MAX_VIDEO_LENGTH_MINUTES = 20
SUMMARY_WORDS = 200


def format_transcript_as_raw_text(transcript, max_video_length_minutes):
    max_length = 60*max_video_length_minutes
    content = []

    last_timestamp = transcript[-1]["start"]
    if last_timestamp > max_length:
        print(f"> Video is longer than {max_video_length_minutes} minutes, unsuitable for GPT. Aborting")
        return ""

    for line in transcript:
        # remove lines with just music or other non-speech
        if (line["text"].startswith("[") and line["text"].endswith("]")):
            continue

        content.append(line["text"])

    return "\n".join(content)


def save_content_to_file(content: str, filename: str):
    with open(f"{filename}.txt", "w", encoding="utf-8") as file:
        file.write(content)


def summarize_transcript(transcript: str):
    print("> Summarizing transcript")

    # no randomness
    temperature = 0

    prompt = f"""
Generate a summary of at minimum {SUMMARY_WORDS} words from the content below, delimited by triple @ symbols.

Content: @@@{transcript}@@@
"""

    # for big texts, use "gpt-3.5-turbo-16k"
    # model = "gpt-3.5-turbo"
    model = "gpt-3.5-turbo-16k"
    messages = [{"role": "user", "content": prompt}]
    completion = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    result = completion.choices[0].message.content

    return result


def download_english_transcript(video_id):
    print(f"> Downloading transcript of video {video_id}")

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    except NoTranscriptFound:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en-US"])

    return transcript


if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY

    video_id = sys.argv[1]
    if not video_id:
        raise Exception("No Youtube video id provided")
    if video_id.startswith("https://www.youtube.com/watch"):
        video_id = video_id.split("v=")[1].split("&")[0]

    if len(sys.argv) == 3:
        max_video_length_minutes = int(sys.argv[2])
    else:
        max_video_length_minutes = MAX_VIDEO_LENGTH_MINUTES

    transcript = download_english_transcript(video_id)
    transcript = format_transcript_as_raw_text(transcript, max_video_length_minutes)
    if len(transcript) > 0:
        save_content_to_file(transcript, f"{video_id}_transcript")
        summary = summarize_transcript(transcript)
        save_content_to_file(summary, f"{video_id}_summary")
        print("> Done")
        print(f"> Summary:\n{summary}")
