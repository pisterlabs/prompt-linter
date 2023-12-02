from langchain.document_loaders import BiliBiliLoader
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser #, OpenAIWhisperParserLocal
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import AZLyricsLoader

from repolya._const import WORKSPACE_TOOLSET
from repolya._log import logger_toolset
import scrapetube
import os


##### BiliBiliLoader
def load_bilibili_transcript_to_docs(video_urls: str):
    loader = BiliBiliLoader(video_urls)
    docs = loader.load()
    return docs


##### YoutubeLoader
def load_youtube_transcript_to_docs(youtube_url: str):
    loader = YoutubeLoader.from_youtube_url(
        youtube_url=youtube_url,
        add_video_info=True,
        language=["en"], # Language param : It's a list of language codes in a descending priority, en by default.
        translation="en", # translation param : It's a translate preference when the youtube does'nt have your select language, en by default.
    )
    docs = loader.load()
    return docs


##### GenericLoader + YoutubeAudioLoader
def load_youtube_audio_to_docs(urls: list[str], save_dir: str):
    loader = GenericLoader(
        YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser()
    )
    docs = loader.load()
    return docs


##### AZLyricsLoader
def load_azlyrics_to_docs(web_paths: list[str]):
    loader = AZLyricsLoader(
        web_paths=web_paths,
        requests_per_second=2,
    )
    docs = loader.aload()
    return docs


##### get_channel_video_urls
def get_channel_video_urls(channel_id='UCvKRFNawVcuz4b9ihUTApCg'):
    _out_dp = str(WORKSPACE_TOOLSET / 'youtube_transcripts' / channel_id)
    if not os.path.exists(_out_dp):
        os.makedirs(_out_dp)
    videos = scrapetube.get_channel(channel_id)
    logger_toolset.info(f"videos: {videos}")
    _video_urls = []
    for video in videos:
        _id = video['videoId']
        _title = video['title']['runs'][0]['text']
        _label = video['title']['accessibility']['accessibilityData']['label']
        print(f"{_id}\n{_title}\n{_label}\n")
        _video_urls.append(f"https://www.youtube.com/watch?v={_id}")
    return _video_urls

