#!/usr/bin/env python3

from typing import Optional, Tuple
import difflib
import multiprocessing
import re
import os
import time

import bs4
import click
import openai
import openai.error
import openai.openai_object
import requests
import youtube_transcript_api

SECTOR_LENGTH = 330
OVERLAP_LENGTH = 30
DEBUG_MERGES = False

MODEL = "gpt-4"
SYSTEM_PROMPT = """Clean up the transcript the user gives you, fixing spelling errors and adding punctuation as needed.
However, do not reword any sentences. Include paragraph breaks where appropriate.""".replace("\n", " ")

def ask_gpt(model: str, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            if isinstance(response, openai.openai_object.OpenAIObject):
                return response.choices[0]["message"]["content"]
            else:
                print(f"Unexpected response from OpenAI: {response}")
                time.sleep(30)
        except openai.error.RateLimitError:
            print("Rate limit error, waiting 30 seconds")
            time.sleep(30)


def clean_transcript_sector(transcript: str) -> str:
    return ask_gpt(MODEL, SYSTEM_PROMPT, transcript)

def merge_similar_strings(s1: str, s1_cutpoint: int, s2: str) -> str:
    # To find the best cutpoint in `s2` for the transition point, we run
    # sequence matching and then figure out where the character corresponding to
    # the one at `s1_cutpoint` in `s1` is in `s2` (erring on the side of
    # including more if it got deleted.)
    matcher = difflib.SequenceMatcher(None, s1, s2)

    s2_cutpoint = None
    previous_s1_block_end = 0
    previous_s2_block_end = 0
    for (i, j, n) in matcher.get_matching_blocks():
        if previous_s1_block_end <= s1_cutpoint < i:
            s2_cutpoint = previous_s2_block_end
            break
        if i <= s1_cutpoint < i + n:
            s2_cutpoint = j + (s1_cutpoint - i)
            break
        previous_s1_block_end = i + n
        previous_s2_block_end = j + n

    if DEBUG_MERGES:
        print(f'Merged: {s1[:s1_cutpoint]}|{s2[s2_cutpoint:]}')

    return s1[:s1_cutpoint] + s2[s2_cutpoint:]

def merge_sectors(sector1, sector2, overlap_chars):
    s1 = sector1[-overlap_chars:]
    s2 = sector2[:overlap_chars]
    merged_portion = merge_similar_strings(s1, overlap_chars // 2, s2)

    return sector1[:-overlap_chars] + merged_portion + sector2[overlap_chars:]

def clean_transcript(transcript: str) -> str:
    transcript_words = re.split(r'\s+', transcript)
    # First we split the transcript into overlapping sectors:
    sectors = []
    sector_starts = []
    for i in range(0, len(transcript_words), SECTOR_LENGTH - OVERLAP_LENGTH):
        sector_starts.append(i)
        sectors.append(' '.join(transcript_words[i:i+SECTOR_LENGTH]))

    # Next we cleanup all the sectors in parallel:
    with multiprocessing.Pool(len(sectors)) as pool:
        cleaned_sectors = pool.map(clean_transcript_sector, sectors)

    # Finally we stich the sectors back together, merging the overlapping portions:
    cleaned_transcript = cleaned_sectors[0]
    for i in range(1, len(cleaned_sectors)):
        overlap_chars = len(' '.join(transcript_words[sector_starts[i]:sector_starts[i] + OVERLAP_LENGTH]))
        cleaned_transcript = merge_sectors(cleaned_transcript, cleaned_sectors[i], overlap_chars)

    # Add a trailing newline if there isn't one:
    if not cleaned_transcript.endswith('\n'):
        cleaned_transcript += '\n'
    return cleaned_transcript

def get_title_and_author_for_video(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    response = requests.get(video_url)
    if response.status_code != 200:
        print(f"Couldn't fetch video page to deduce title and author: {video_url}")
        return (None, None)
    page_text = requests.get(video_url).text
    soup = bs4.BeautifulSoup(page_text, 'html.parser')

    title_tag = soup.find('title')
    title = None
    if title_tag:
        title = title_tag.text
        title = title.replace(" - YouTube", "")

    author = None
    author_tag = soup.find('link', {'itemprop': 'name'})
    if isinstance(author_tag, bs4.element.Tag):
        author = author_tag.attrs.get('content', None)
    return (title, author)

def extract_video_id(url_or_id: str) -> str:
    url_or_id = url_or_id.replace('\\', '')
    m = re.search(r'v=([a-zA-Z0-9_-]+)', url_or_id)
    if m:
        return m.group(1)
    else:
        return url_or_id

@click.command()
@click.argument('video')
@click.argument('output_file')
def main(video: str, output_file: str):
    video_id = extract_video_id(video)

    # First let's extract get the title and author from the video page:
    title, author = get_title_and_author_for_video(video_id)

    # Grab the transcript:
    transcript = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(video_id)
    original_text = ' '.join(word for line in transcript for word in line['text'].split())

    cleaned_text = clean_transcript(original_text)

    with open(output_file, 'w') as f:
        if title:
            f.write("Video Title: " + title + "\n")
        if author:
            f.write("Video Author: " + author + "\n")
        f.write("\n")
        f.write(cleaned_text)



if __name__ == "__main__":
    main()