import webvtt
from openai import OpenAI
import os

client = OpenAI()
MODEL_NAME = 'gpt-4-1106-preview'


def extract_text_from_vtt(vtt_file_path):
    vtt = webvtt.read(vtt_file_path)
    text = ""
    for line in vtt:
        text += line.text
    return text


def summarize(text):
    system_prompt = """You are a professional summarizer. You are given a transcript of a video and you need to summarize it.
    When you summarize, please keep the following points in mind.
    - Please include important points/announcements/examples in the video.

    Please summarize the transcipt in following format.
    # Summary
    - {topic name}
    {summary in 2-5 sentences}
    - {topic name}
    {summary in 2-5 sentences}
    (Repeat until you summarize all the topics)
    """

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        model=MODEL_NAME,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def translate(text):
    system_prompt = "You are a professional translator. Please translate the given text into fluent Japanese. If given text is in Japanese, just return the text as it is."
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        model=MODEL_NAME,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


if __name__ == '__main__':
    video_id = 'FZhbJZEgKQ4'
    vtt_file_path = f'{video_id}.en.vtt'
    text = extract_text_from_vtt(vtt_file_path)
    summary = summarize(text)
    with open(f"data/{video_id}_summary.txt", "w") as file:
        file.write(summary)
    summary_ja = translate(summary)
    with open(f"data/{video_id}_summary_ja.txt", "w") as file:
        file.write(summary_ja)
