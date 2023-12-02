import os
import re
from datetime import datetime, timedelta

import googleapiclient.discovery
from dotenv import load_dotenv
from google.oauth2 import service_account
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders.base import Document

load_dotenv()

# Set up the API client
api_key = os.getenv("GOOGLE_API_KEY")
service = googleapiclient.discovery.build(
    "youtube", "v3", developerKey=api_key
)


def get_transcript(video_id: str) -> list[Document]:
    """Get the transcript of a YouTube video

    Arguments:
        video_id {str} -- The ID of the YouTube video

    Returns:
        list[Document] -- The transcript of the YouTube video in langchain Document format
    """
    loader = YoutubeLoader.from_youtube_url(
        f"https://www.youtube.com/watch?v={video_id}",
        language="en",
    )
    transcript = loader.load()
    return transcript


def get_videos(channel_id: str, videos: list[dict[str, str]]) -> None:
    """Get the videos from a YouTube channel in the last 24 hours

    Arguments:
        channel_id {str} -- The ID of the YouTube channel
        videos {list[dict[str, str]]} -- The list to append the new videos to
    """
    one_day_ago = (datetime.now() - timedelta(days=1)).isoformat() + "Z"

    request = service.search().list(
        channelId=channel_id,
        part="snippet",
        type="video",
        order="date",
        publishedAfter=one_day_ago,
    )

    response = request.execute()

    for item in response["items"]:
        video_title = item["snippet"]["title"]
        video_id = item["id"]["videoId"]
        videos.append({"title": video_title, "id": video_id})


def get_channel_id(channel_url: str) -> str | None:
    """Get the ID of a YouTube channel from the URL

    Arguments:
        channel_url {str} -- The URL of the YouTube channel
    """
    match = re.search(r"/@([\w-]+)/", channel_url)
    if match:
        username = match.group(1)
        search_response = (
            service.search()
            .list(
                q=username,
                type="channel",
                part="id",
                maxResults=1,
            )
            .execute()
        )

        channel_id = search_response["items"][0]["id"]["channelId"]

        return channel_id
    return None


def get_videos_yesterday() -> str:
    """Get the videos from the last 24 hours from the channels in channel_urls

    Returns:
        str -- The videos from the last 24 hours as a string
    """
    channel_urls = [
        "https://www.youtube.com/@MattVidPro/videos",
        "https://www.youtube.com/@airevolutionx/videos",
        "https://www.youtube.com/@aiexplained-official/videos",
        "https://www.youtube.com/@matthew_berman/videos",
        "https://www.youtube.com/@mreflow/videos",
        "https://www.youtube.com/@1littlecoder/videos",
    ]

    videos = []
    for url in channel_urls:
        channel_id = get_channel_id(url)
        get_videos(channel_id, videos)

    corpora = ""
    docs = []
    for video in videos:
        doc = get_transcript(video["id"])
        docs.append(doc[0].page_content)
        corpora += doc[0].page_content + ". "

    return corpora
