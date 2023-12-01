from configparser import InterpolationMissingOptionError
import sys
import os
import time
from typing import final
import openai
import tiktoken
import yt_dlp as youtube_dl
from youtube_transcript_api import YouTubeTranscriptApi
import nltk

MAX_INTERIM_WORD_LENGTH = 250
MAX_FINAL_WORD_LENGTH = 1000
CHUNK_SIZE = 250
MODEL="gpt-3.5-turbo"

encoding = tiktoken.encoding_for_model(MODEL)

interim_prompt="""
You will be provided with a portion of a transcript of a YouTube video and your task is to summarize the text in {{maxwords}} words or less,
also considering the context given before and after it.  Only provide the final summary and do not include the context in your response.
Context before:{{before}}
Portion of transcript to summarize:{{current}}
Context after:{{after}}

"""

final_prompt="""
You will be provided with a transcript for a video titled {{title}} and your task is to summarize the transcript.
Your output should use the following template:
### Summary
### Notes
- Bulletpoint
### Keywords
- Explanation
Create 10 bullet points that summarize the key points.
Also extract the important keywords and for each keyword provide an explanation and definition based on its occurrence in the transcription.
Ensure your response is less than {{maxwords}} words.
Here is the transcript: {{transcript}}
"""

# Define function to read OpenAI API key from file
def read_api_key():
    api_key_file = os.path.expanduser("~/.openai-api-key.txt")
    if not os.path.exists(api_key_file):
        print("OpenAI API key file not found.")
        sys.exit(1)
    with open(api_key_file, "r") as f:
        api_key = f.read().strip()
    return api_key

# Set up OpenAI API key
openai.api_key = read_api_key()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

# Define function to download transcript from YouTube video
def download_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = ""
    for line in transcript_list:
        transcript_text += line['text'] + " "
    return transcript_text

class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        #print(msg)
        pass

def get_title(video_id):
    ydl_opts = {}

    url = f"https://www.youtube.com/watch?v={video_id}"

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        return info_dict.get('title', None)

def chunk_sentences(text, max_words=CHUNK_SIZE):
    chunks = []
    current_chunk_words = 0
    current_chunk = ""

    sentences = nltk.sent_tokenize(text)

    # If # of sentences is 1, it means there is no punctuation in the text.
    # In this case, we will split the text into chunks based on the max_words
    if len(sentences) == 1:
        words = text.split()
        for word in words:
            current_chunk += word + " "
            current_chunk_words += 1
            if current_chunk_words >= max_words:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_chunk_words = 0
        if current_chunk:
            chunks.append(current_chunk.strip())
    else:
        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding the current sentence to the current chunk would exceed the max_words
            # then start a new chunk.
            if current_chunk_words + sentence_words > max_words:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_chunk_words = 0

            current_chunk += sentence + " "
            current_chunk_words += sentence_words

        # Append any remaining text
        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks

def delayed_completion(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return openai.ChatCompletion.create(**kwargs)

def chat_with_gpt(title, transcript_text):
    responses = []
    chunks = []

    # Calculate the delay based on your rate limit
    rate_limit_per_minute = 20
    delay = 60.0 / rate_limit_per_minute

    chunks = chunk_sentences(transcript_text, CHUNK_SIZE)
    numchunks = len(chunks)
    curr_chunk = 0

    # Process each chunk through GPT-4
    for chunk in chunks:
        print(f"Processing chunk {curr_chunk+1} of {numchunks}...")

        if curr_chunk == 1:
            before = ""
        else:
            before = chunks[curr_chunk-1]

        if curr_chunk == numchunks-1:
            after = ""
        else:
            after = chunks[curr_chunk+1]

        updated_interim_prompt = interim_prompt.replace("{{before}}", before).replace("{{current}}", chunk).replace("{{after}}", after).replace("{{maxwords}}", str(MAX_INTERIM_WORD_LENGTH))

        messages=[
                {"role": "user", "content": updated_interim_prompt}
        ]
        chunk_word_count = len(chunk.split())
        print(f"   Chunk word count: {chunk_word_count}")
        print(f"   Chunk token count: {num_tokens_from_messages(messages, model=MODEL)}")

        response = delayed_completion(
            model=MODEL,
            messages=messages,
            max_tokens=3000
        )
        print("done.")
        interim = response.choices[0].message.content
        responses.append(interim)
        curr_chunk += 1

    all_responses = ' '.join(responses)
    print(f"Total word count: {len(all_responses.split())}")

    # Now process the summary through GPT-4 using the final prompt
    print("Processing final prompt...")
    updated_final_prompt = final_prompt.replace("{{title}}", title).replace("{{transcript}}", all_responses).replace("{{maxwords}}", str(MAX_FINAL_WORD_LENGTH))
    messages=[
            {"role": "user", "content": updated_final_prompt}
    ]
    final_word_count = len(all_responses.split())
    print(f"   Final word count: {final_word_count}")
    print(f"   Final token count: {num_tokens_from_messages(messages, model=MODEL)}")
    response = delayed_completion(
            model="gpt-4",
            messages=messages
        )

    return response.choices[0].message.content

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a YouTube video ID as an argument.")
        sys.exit(1)
    video_id = sys.argv[1]
    print(f"Processing video ID: {video_id}")
    title = get_title(video_id)
    print(f"Video title: {title}")
    print("Downloading transcript...", end=" ")
    transcript_text = download_transcript(video_id)
    print("done.")
    wordcount = len(transcript_text.split())
    print(f"Transcript word count: {wordcount}")
    summary = chat_with_gpt(title, transcript_text)
    print("Results:\n")
    print(summary)
