#!/usr/bin/env python3

from yt_dlp import YoutubeDL
import re
import argparse
import os
import openai
import textwrap
import json
import time

# Get OpenAI API key from environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define a variable to keep track of total tokens
total_tokens = 0

def clear_log_directory():
    # Check if the directory exists
    if not os.path.exists('log'):
        return

    # Get a list of all files in the directory
    files = os.listdir('log')

    print(f"Deleting {len(files)} files from the log directory...")
    # Loop through the files and delete each one
    for file_name in files:
        os.remove(os.path.join('log', file_name))

def write_response_to_file(response, count):
    # Create a directory called 'log' if it doesn't exist
    if not os.path.exists('log'):
        os.makedirs('log')

    # Create a file name inside the 'log' directory
    file_name = os.path.join('log', f"response_{count}.json")

    # Write the response to a file inside the 'log' directory
    with open(file_name, 'w') as f:
        json.dump(response, f)

def download_youtube_subtitle(url):
    import io

    # Set YoutubeDL options
    ydl_opts = {
        'writesubtitles': True,    # Download subtitles
        'skip_download': True,     # Skip downloading the video
        'writeautomaticsub': True, # Write automatic subtitles
        'quiet': True,             # Keep quiet
        'convertsubtitles': 'srt', # Convert the subtitles to srt format
        'subtitleslangs': ['en'],  # Download English subtitles only
        'outtmpl': 'output.srt'    # Name the subtitles file 'output.srt'
    }

    # Create a YoutubeDL instance with the specified options
    with YoutubeDL(ydl_opts) as ydl:
        # Download the subtitles and video information for the given URL
        info_dict = ydl.extract_info(url, download=False)
        title = info_dict.get('title', None)
        ydl.download([url])

    # Open the file containing the text
    with open("output.srt.en.vtt", "r") as f:
        text = f.read()

    # Remove the metadata and timecodes using regular expressions
    clean_text = re.sub(r"<.*?>", "", text)
    clean_text = re.sub(r".*align:start position:.*\n", "", clean_text)

    # Remove empty lines
    clean_text = "\n".join([line for line in clean_text.split("\n") if line.strip()])

    # Write the cleaned SRT text to a string buffer
    cleaned_srt = io.StringIO()
    for line in clean_text.split("\n"):
        cleaned_srt.write(line)
        cleaned_srt.write("\n")
    # Remove the temporary files
    os.remove("output.srt.en.vtt")
    # Return the title of the video and the cleaned SRT text
    return title, cleaned_srt.getvalue()

# Define function to break text into chunks
def chunk_text(text, chunk_size):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Define function to summarize a chunk of text
def summarize_chunk(chunk, count, debugging):
    global total_tokens # Use the global total_tokens variable
    start_time = time.time() # Start the timer
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            { "role": "system", "content": "" },
            { "role": "user", "content": chunk },
        ],
        temperature=0.7,
    )
    end_time = time.time() # Stop the timer
    elapsed_time_ms = int((end_time - start_time) * 1000) # Calculate the elapsed time in ms
    token_count = response['usage']['total_tokens']
    total_tokens += token_count # Update the total token count

    print(f"Sent request {count} to OpenAI API ({elapsed_time_ms}ms) ({total_tokens} tokens)...")
    if debugging:
        # Write the response to a file
        write_response_to_file(response, count)
    summary = response['choices'][0]['message']['content']
    return summary.lstrip()

# Define function to rewrite a text using OpenAI
def rewrite_text(text, debugging):
    global total_tokens # use the global total_tokens variable
    start_time = time.time() # Start the timer
    response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
        messages=[
            { "role": "system", "content": "" },
            { "role": "user", "content": (
                    f"please provide a brief summary of this video:\n{text}\n"), },
        ],
        temperature=0.7,
    )
    end_time = time.time() # Stop the timer
    elapsed_time_ms = int((end_time - start_time) * 1000) # Calculate the elapsed time in ms
    token_count = response['usage']['total_tokens']
    total_tokens += token_count # Update the total token count

    print(f"Sent rewrite request to OpenAI API ({elapsed_time_ms}ms) ({total_tokens} tokens)...")
    if debugging:
        # Write the response to a file
        write_response_to_file(response, "rewrite")
    rewritten_text = response['choices'][0]['message']['content']
    return rewritten_text.lstrip()

# Define function to summarize a full text file
def summarize_file(text, chunk_size, debugging):
    # Clean up the text
    text = re.sub("\n+", "\n", text)
    text = re.sub("\n", ". ", text)
    text = re.sub(" +", " ", text)
    text = text.strip()

    # Break the text into chunks
    chunks = chunk_text(text, chunk_size)

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarize_chunk(chunk, i+1, debugging)
        summaries.append(summary)

    # Join the summaries together into a single text
    summary = " ".join(summaries)
    rewritten_summary = rewrite_text(summary, debugging)
    wrapped_summary = textwrap.fill(rewritten_summary, width=80)
    wrapped_summary = wrapped_summary.strip()
    #return rewritten_summary 
    return wrapped_summary.lstrip()

# Define function to suggest hashtags based on input text
def suggest_hashtags(text, debugging):
    global total_tokens # use the global total_tokens variable
    start_time = time.time() # Start the timer
    response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
        messages=[
            { "role": "system", "content": "" },
            { "role": "user", "content": (
                    f"Please suggest 5 hashtags for this text:\n{text}\n"
                     "Hashtags: "), },
        ],
        temperature=0.7,
    )
    end_time = time.time() # Stop the timer
    elapsed_time_ms = int((end_time - start_time) * 1000) # Calculate the elapsed time in ms
    token_count = response['usage']['total_tokens']
    total_tokens += token_count # Update the total token count

    print(f"Sent hashtag request to OpenAI API ({elapsed_time_ms}ms) ({total_tokens} tokens)...")
    if debugging:
        # Write the response to a file
        write_response_to_file(response, "hashtag_suggestion")
    hashtags = response['choices'][0]['message']['content']
    return hashtags.lstrip()

def suggest_title(text, debugging):
    global total_tokens # use the global total_tokens variable
    start_time = time.time() # Start the timer
    response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
        messages=[
            { "role": "system", "content": "" },
            { "role": "user", "content": (
                    f"Please suggest a short title for the video described in this text:\n{text}\n"), },
        ],
        temperature=0.7,
    )
    end_time = time.time() # Stop the timer
    elapsed_time_ms = int((end_time - start_time) * 1000) # Calculate the elapsed time in ms
    token_count = response['usage']['total_tokens']
    total_tokens += token_count # Update the total token count

    print(f"Sent title request to OpenAI API ({elapsed_time_ms}ms) ({total_tokens} tokens)...")
    if debugging:
        # Write the response to a file
        write_response_to_file(response, "title_suggestion")
    video_title = response['choices'][0]['message']['content']
    return video_title.lstrip()

def get_openai_api_cost(num_tokens):
    cost_per_token = 0.0002
    total_cost = num_tokens * cost_per_token
    rounded_cost = round(total_cost / 10, 2)
    return rounded_cost

# Define main function
def main(chunk_size, debugging):
    global total_tokens # Use the global total_tokens variable
    # If --debugging option is present, clear the log directory
    if args.debugging:
        clear_log_directory()

    # Download subtitles for the given video URL
    title, text = download_youtube_subtitle(args.url)
    #print(f"Downloaded subtitles for '{title}'\n\n")
    summary = summarize_file(text, chunk_size, debugging)
    hashtags = suggest_hashtags(summary, debugging)
    video_title = suggest_title(summary, debugging)
    total_cost = get_openai_api_cost(total_tokens)

    print(f"Total tokens used: {total_tokens} (Cost: {total_cost})\n\n")
    print("\n\nSuggested Title: ", video_title, "\n\nSuggested Summary:\n", summary, "\n\n")
    print("Suggested Hashtags: \n", hashtags, "\n\n")


# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument("--chunk_size", type=int, default=5000, help="size of each text chunk (default: 5000)")
    parser.add_argument("--debugging", action="store_true", help="enable debugging mode")
    args = parser.parse_args()

    # Call the main function with the given arguments
    main(args.chunk_size, args.debugging)
