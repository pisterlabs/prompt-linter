import os
import re

import openai
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# Load the environment variables from the .env file
load_dotenv()


def extract_youtube_video_id(url: str) -> str | None:
    """
    Extract the video ID from the URL
    https://www.youtube.com/watch?v=XXX -> XXX
    https://youtu.be/XXX -> XXX
    """
    found = re.search(r"(?:youtu\.be\/|watch\?v=)([\w-]+)", url)
    if found:
        return found.group(1)
    return None


def get_video_transcript(video_id: str) -> str | None:
    """
    Fetch the transcript of the provided YouTube video
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        # The video doesn't have a transcript
        return None

    text = " ".join([line["text"] for line in transcript])
    return text


def generate_summary(text: str) -> str:
    """
    Generate a summary of the provided text using OpenAI API
    """
    # Initialize the OpenAI API client
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Use GPT to generate a summary
    instructions = "Please summarize the provided text"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        n=1,
        max_tokens=200,
        presence_penalty=0,
        frequency_penalty=0.1,
    )

    # Return the generated summary
    return response.choices[0].message.content.strip()


def summarize_youtube_video(video_url: str) -> str:
    """
    Summarize the provided YouTube video
    """
    # Extract the video ID from the URL
    video_id = extract_youtube_video_id(video_url)

    # Fetch the video transcript
    transcript = get_video_transcript(video_id)

    # If no transcript is found, return an error message
    if not transcript:
        return f"No English transcript found " \
               f"for this video: {video_url}"

    # Generate the summary
    summary = generate_summary(transcript)

    # Return the summary
    return summary


if __name__ == '__main__':
    url = "https://www.youtube.com/watch?v=D1R-jKKp3NA"
    print(summarize_youtube_video(url))
