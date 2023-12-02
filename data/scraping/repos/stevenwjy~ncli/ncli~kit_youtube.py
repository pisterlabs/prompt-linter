"""
A module for processing and managing YouTube data.
"""

import os

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

from pydantic import BaseModel  # pylint: disable=no-name-in-module

import openai
import pytube
from click import echo
from youtube_transcript_api import YouTubeTranscriptApi

from ncli.utils import format_duration


class Config(BaseModel):
    """
    Config for YouTube operations.
    """
    export_dir: Optional[str] = None

    language: str = 'en'

    # Defaults to 15 mins, which is typically near borderline to fit within 8K context length.
    summary_time_window_minutes: int = 15

    # Should be fine to use 'gpt-3.5-turbo', which is more accessible (not under limited beta) and is much cheaper
    # at the moment. For slightly better result, use 'gpt-4'. If you need longer context (e.g., because of longer
    # time window), you can also use `gpt-4-32k`.
    #
    # Note that you need `OPENAI_API_KEY` env var to access OpenAI API.
    #
    # If the model is set to an empty string, the summary will just be a concatenation of the texts to
    # be summarized. This may be useful for those who want to do the summarization using external app
    # (e.g., if already have ChatGPT Plus subscription).
    model: str = ''

    prompt_system: str = 'You are a helpful assistant.'

    prompt_summarize: str = \
        'Can you analyze all the key points and arrange them into cohesive paragraphs? ' \
        'Do not reorder/remove any of the key points. Keep the number of paragraphs to a minimum. ' \
        'Respond only with the paragraphs. Do not add your own words.'


@dataclass
class TranscriptItem:
    """
    A class that represents a single transcript item in a video.

    Attributes:
    start_ts (float): Start timestamp in seconds from the beginning of the video.
    duration (float): Duration of the transcript item in seconds.
    text (str): The actual transcript text.
    """
    start_ts: float
    duration: float
    text: str


@dataclass
class Transcript:
    """
    A class that represents a transcript of a video.

    Attributes:
    items (List[TranscriptItem]): List of transcript items.
    language (str): Language of the transcript, default is 'en' (English).
    """
    items: List[TranscriptItem]
    language: str = 'en'


@dataclass
class VideoData:
    """
    A class that represents metadata and transcripts of a video.

    Attributes:
    title (str): The title of the video.
    url (str): The URL of the video.
    author (str): The author of the video.
    length (float): The length of the video in seconds.
    publish_date (datetime): The date when the video was published.
    keywords (List[str]): List of keywords relevant to the video.
    description (Optional[str]): The description of the video. It can be None.
    transcript (Transcript): The transcript of the video.
    summary (Optional[Transcript]): The summary of the video in form of a Transcript object. It can be None.

    Note:
    Rating and view count are not captured intentionally as they can change continuously and be distracting.
    To see the latest state, open the original url.
    """
    title: str
    url: str
    author: str
    length: float  # in seconds
    publish_date: datetime
    keywords: List[str]
    description: Optional[str]
    transcript: Transcript
    summary: Optional[Transcript] = None


def _get_transcript(video_id: str, language='en') -> Transcript:
    transcript = YouTubeTranscriptApi.get_transcript(
        video_id, languages=(language,))

    items: List[TranscriptItem] = []
    for part in transcript:
        # split() without arguments splits the string at whitespace groups (spaces, tabs, newlines),
        # and join() concatenates them together with a single space.
        text = ' '.join(part['text'].split())

        items.append(TranscriptItem(
            start_ts=part['start'],
            duration=part['duration'],
            text=text,
        ))

    transcript = Transcript(items=items, language=language)
    return transcript


def _summarize(transcript: Transcript, config: Config) -> Transcript:
    summary_items = []

    cur_group_start_ts = None
    cur_group_end_ts = None
    cur_group_texts = []

    for item in transcript.items:
        within_cur_group = cur_group_start_ts is not None and \
            item.start_ts - cur_group_start_ts < config.summary_time_window_minutes * 60

        if within_cur_group:
            cur_group_end_ts = item.start_ts + item.duration
            cur_group_texts.append(item.text)
        else:
            # If there's an existing group, add as summary
            if cur_group_start_ts is not None:
                summary_items.append(_create_transcript_summary_item(
                    cur_group_start_ts, cur_group_end_ts, cur_group_texts, config))

            # Create new group
            cur_group_start_ts = item.start_ts
            cur_group_end_ts = item.start_ts + item.duration
            cur_group_texts = [item.text]

    # Last group
    if cur_group_start_ts is not None:
        summary_items.append(_create_transcript_summary_item(
            cur_group_start_ts, cur_group_end_ts, cur_group_texts, config))

    return Transcript(items=summary_items)


def _create_transcript_summary_item(
    start_ts: float,
    end_ts: float,
    texts: List[str],
    config: Config,
) -> TranscriptItem:
    text = ' '.join(texts)

    if config.model:
        # Prereq: set OPENAI_API_KEY env var
        try:
            response = openai.ChatCompletion.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": config.prompt_system},
                    {"role": "user", "content": f'Transcript:\n"""\n{text}\n"""\n\n{config.prompt_summarize}'},
                ]
            )
            text = response['choices'][0]['message']['content']

            # Metadata for tracking/debugging
            response_metadata = {
                'id': response['id'],
                'model': response['model'],
                'object': response['object'],
                'usage': response['usage'],
                'finish_reason': response['choices'][0]['finish_reason'],
            }
            echo(f'OpenAI metadata: {response_metadata}')
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Avoid failing the entire summarization just because a window fails to be summarized
            # (e.g., issue from the OpenAI API) or the response format changes for the metadata.
            #
            # Note that we may either end up with just concatenated transcript (if the request failed),
            # or still have the text being summarized properly (if it failed when parsing the response metadata).
            echo(f'Failed to retrieve response properly. Reason: {e}')

    return TranscriptItem(start_ts=start_ts, duration=end_ts-start_ts, text=text)


def _extract_video_id(url: str):
    parsed_url = urlparse(url)
    if parsed_url.netloc == "youtu.be":
        return parsed_url.path[1:]
    if parsed_url.netloc in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            p = parse_qs(parsed_url.query)
            return p['v'][0]
        if parsed_url.path[:7] == "/embed/":
            return parsed_url.path.split("/")[2]
        if parsed_url.path[:3] == "/v/":
            return parsed_url.path.split("/")[2]
    return None


def _extract_video_data(
    video_url: str,
    config: Config,
    with_summary=False,
) -> VideoData:
    data = pytube.YouTube(video_url)

    video_id = _extract_video_id(video_url)
    transcript = _get_transcript(video_id, config.language)

    summary = None
    if with_summary:
        summary = _summarize(transcript, config)

    return VideoData(
        title=data.title,
        url=video_url,
        author=data.author,
        length=data.length,
        publish_date=data.publish_date,
        keywords=data.keywords,
        description=data.description,
        transcript=transcript,
        summary=summary,
    )


def export(
    video_url: str,
    target_dir: Path,
    with_transcript: bool,
    with_summary: bool,
    config: Config,
):
    """
    Exports the video data to a markdown file in the target directory.

    Parameters:
    video_url (str): The URL of the video to be exported.
    target_dir (Path): The directory where the markdown file should be saved.
    with_summary (bool): If True, include the summary of the video in the markdown file.
    config (Config): The configuration object containing necessary parameters to export the data.
    """
    os.makedirs(target_dir, exist_ok=True)
    video = _extract_video_data(video_url, config, with_summary=with_summary)

    output_file = target_dir.joinpath(f'{video.title}.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'# {video.title}\n\n')

        f.write(f'- URL: {video.url}\n')
        f.write(f'- Author: {video.author}\n')
        f.write(f'- Length: {format_duration(video.length)}\n')
        f.write(f'- Publish date: {video.publish_date.strftime("%Y-%m-%d")}\n')
        if video.keywords:
            f.write(f'- Keywords: {", ".join(video.keywords)}\n')
        if video.description:
            f.write('\n**Description:**\n')
            f.write(f'\n{video.description}\n')

        if video.summary:
            f.write('\n## Summary\n')
            for item in video.summary.items:
                # Note that we use more than one newlines at the end of the timestamp to put the summary text
                # as a separate line, given that it could sometimes consist of multiple paragraphs.
                f.write(f'\n[{format_duration(item.start_ts)}]\n\n')
                f.write(f'{item.text}\n')

        if with_transcript:
            f.write('\n## Transcript\n')
            for item in video.transcript.items:
                f.write(f'\n[{format_duration(item.start_ts)}]\n')
                f.write(f'{item.text}\n')
