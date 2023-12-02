import os
import pathlib

import openai
from youtube_lib import youtube_videos
from youtube_transcript_api import YouTubeTranscriptApi

CHARS_PER_TOKEN = 4
MAX_CHARS = 4000 * 4
MAX_PROMPT = int(0.75 * MAX_CHARS)


ANKI_PROMPT = """Please help me create Anki cards for some material that I am studying. I would like to use these cards to remember facts and information in this material. Each Anki card should be of the format:

{FRONT} ; {BACK}

Where {FRONT} is the front of the card, and {BACK} is the back of the card, and they are separated by a semi-colon. The way it works is that Anki will show me the {FRONT} of the card, which contains some kind of question, and I will have to correctly recall the {BACK} of the card. Please give me the Anki cards one per line so it is easy for me to copy paste and import them into Anki. Make sure to be thorough and cover most of the information in the given material. Here are some examples of good Anki cards. ONLY output lines like the following.

What is the capital city of California? ; Sacramento
How many U.S. states are there? ; 50
What is the smallest U.S. state? ; Wyoming

Etc. Now here is the material I’d like you to create Anki cards for:
"""


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


def get_cards(text):
    prompt = ANKI_PROMPT + text
    print(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response


if __name__ == "__main__":
    artifact_dir = pathlib.Path("files")
    artifact_dir.mkdir(exist_ok=True)
    for title, video_id in youtube_videos():
        print(title, video_id)

        # Extract transcript. Skip processing if already exists
        transcription = YouTubeTranscriptApi.get_transcript(video_id)
        transcription_text = " ".join([t["text"] for t in transcription])
        transcription_filepath = artifact_dir / f"{title}_transcription.txt"
        if transcription_filepath.exists():
            print("Transcript exists, skipping...")
            continue
        with open(transcription_filepath, "w") as f:
            f.write(transcription_text)

        # Generate
        for chunk_start in range(0, len(transcription_text), MAX_PROMPT):
            idx = chunk_start // MAX_PROMPT
            text = transcription_text[chunk_start : chunk_start + MAX_PROMPT]
            summary_filepath = artifact_dir / f"{title}_summary_{idx}.txt"
            resp = summarize(text)
            with open(summary_filepath, "w") as f:
                f.write(resp["choices"][0]["message"]["content"])

            anki_filepath = artifact_dir / f"{title}_anki_{idx}.txt"
            resp = get_cards(text)
            with open(anki_filepath, "w") as f:
                f.write(resp["choices"][0]["message"]["content"])
