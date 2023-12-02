from video_dir_helper import get_video_segments
from dotenv import load_dotenv
from datetime import datetime

import time
import re
import openai
import os
import json
import sys
import random
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray

load_dotenv()

# Watch out for OpenAI rate limits
# https://platform.openai.com/account/rate-limits

# Initialize Ray
ray.init()

openai.api_key = os.environ.get("OPENAI_KEY")


def call_chatgpt_api(prompt, api_key, max_retries=5):
    # Set the API key in the worker process
    openai.api_key = api_key

    # Send a chat completion request to the OpenAI API
    retry_count = 0
    while retry_count <= max_retries:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )

            # Introduce a delay before the next API call
            if len(prompt) > 3000:
                time.sleep(3)
            elif len(prompt) > 2000:
                time.sleep(2)
            elif len(prompt) > 1000:
                time.sleep(1)

            # Successful API call, return the response
            return completion.choices[0].message["content"]

        except openai.error.RateLimitError:
            # If we're rate limited, wait and retry
            wait_time = (2**retry_count) + random.random() * 0.01
            print(
                f"Rate limit exceeded, waiting for {wait_time} seconds before retrying..."
            )
            time.sleep(wait_time)
            retry_count += 1

        # If max retries reached, raise an error
        if retry_count > max_retries:
            raise Exception("Max retries reached, aborting...")


def process_srt(srt_text):
    lines = srt_text
    if type(srt_text) == str:
        lines = srt_text.split("\n")

    # Remove duplicates while preserving order
    seen = set()
    lines = [x for x in lines if not (x in seen or seen.add(x))]

    # Remove empty lines, lines with just numbers, and lines with timestamps
    processed_lines = []
    for line in lines:
        # If line is not empty and doesn't match the patterns
        if (
            line
            and not re.match(r"^\s*$", line)
            and not re.match(r"^\d+$", line)
            and not re.match(
                r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$", line
            )
        ):
            processed_lines.append(line)

    # Rejoin lines
    processed_text = "".join(processed_lines)
    return processed_text


@ray.remote
def get_video_title(tot_segment_number, segment_transcript, api_key):
    print(f"Processing: {tot_segment_number}, {segment_transcript}")
    i = 0
    summaries = []
    tot_prompt_len = 0

    with open(segment_transcript, "r") as f:
        lines = f.readlines()

    parsed_lines = process_srt(lines)
    parsed_transcript = segment_transcript.replace(".srt", ".txt")
    with open(parsed_transcript, "w") as f:
        f.write(parsed_lines)

    # one token generally corresponds to ~4 characters of text
    # limit of 4,096 tokens
    # window_size = 3000 tokens * 4 chars / 1 token * 3 lines / 70 chars = 514 lines
    window_size = 500
    parsed_lines = parsed_lines.split("\n")
    while i < len(parsed_lines):
        prompt = (
            "\n".join(parsed_lines[i : i + window_size])
            + "Summarize above. Be concise."
        )
        tot_prompt_len += len(prompt)
        response = call_chatgpt_api(prompt, api_key)
        # print("Summary:", response)
        summaries.append(response)
        i += window_size

    summary = "\n".join(summaries) + "\n"
    summaries_file = segment_transcript.replace(".srt", ".summaries.txt")
    with open(summaries_file, "w") as f:
        f.write(summary)

    prompt = (
        summary
        + "Generate a short YouTube video title up to 5 words from the summaries above."
    )
    tot_prompt_len += len(prompt)
    title_response = call_chatgpt_api(prompt, api_key)
    title_response = title_response.replace('"', "").replace(":", " -")
    # print("Title:", title_response)

    prompt = summary + "Summarize above. Be concise."
    tot_prompt_len += len(prompt)
    summary_response = call_chatgpt_api(prompt, api_key)
    # print("Summary:", summary_response)

    prompt = (
        "\n".join(lines[:100])
        + "The above is a video transcription in srt format. Give the time the speaker starts speaking. "
        + "Give only the time and NO other text. Use the format: 00:01:17,210."
    )
    tot_prompt_len += len(prompt)
    retry_count = 0
    max_retries = 5
    while retry_count <= max_retries:
        start_time_response = call_chatgpt_api(prompt, api_key)
        try:
            match_time = re.search(r"\d{2}:\d{2}:\d{2},\d{3}", start_time_response)
            timestamp = match_time.group()  # this might error. could try a retry
            hours, minutes, seconds_millisec = timestamp.split(":")
            seconds, _ = seconds_millisec.split(",")

            # Convert minute and second to integers
            hours = int(hours)
            minutes = int(minutes)
            seconds = int(seconds)

            match_segment = re.search(r"segment(\d+)", segment_transcript)
            segment_number = int(match_segment.group(1))

            # Extract the minutes and seconds
            minutes = minutes + 60 * hours - 10 * segment_number
            seconds = seconds
            start_time_response = "00:00"
            if minutes >= 0:
                start_time_response = f"{minutes:02d}:{seconds:02d}"
            break
        except Exception:
            print(
                f"Failed to decode time format: {start_time_response} for {segment_transcript}"
            )
            retry_count += 1

        # If max retries reached, raise an error
        if retry_count > max_retries:
            raise Exception("Max retries reached for time parser, aborting...")
        # print("Start Time:", start_time_response)

    output_name = segment_transcript.replace(".srt", ".json")
    with open(output_name, "w") as f:
        json.dump(
            {
                "title": title_response,
                "summary": summary_response,
                "start_time": start_time_response,
            },
            f,
            indent=2,
        )

    print(f"Response written to {tot_segment_number}, {output_name}")

    if tot_prompt_len > 25000:
        time.sleep(25)
    elif tot_prompt_len > 20000:
        time.sleep(20)
    elif tot_prompt_len > 15000:
        time.sleep(15)
    elif tot_prompt_len > 10000:
        time.sleep(10)
    elif tot_prompt_len > 5000:
        time.sleep(5)


segments = get_video_segments()
i = 0
if len(sys.argv) == 2:
    i = int(sys.argv[1])

ray.get(
    [
        get_video_title.remote(index, segment[1], openai.api_key)
        for index, segment in enumerate(segments[i:], start=i)
    ]
)

# Execute Serially
# for j in range(i, len(segments)):
#     segment = segments[j]
#     get_video_title(j, segment[1], openai.api_key)
