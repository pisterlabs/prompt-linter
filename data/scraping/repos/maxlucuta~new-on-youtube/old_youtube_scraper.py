from youtube_transcript_api import YouTubeTranscriptApi
from .gpt3 import summarize_yt_script_with_gpt3
from openai.error import RateLimitError
from openai.error import ServiceUnavailableError
from openai.error import InvalidRequestError
from youtubesearchpython import (
    Video,
    Suggestions,
    VideoDurationFilter,
    VideoSortOrder,
    CustomSearch,
    VideosSearch,
)
from abc import abstractmethod
from .database import db_contains_video as in_db
import youtube_transcript_api
import time
import requests


class YouTubeScraper:
    def __init__(self):
        self.response = []

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def _insert_results(self, result):
        pass

    @staticmethod
    def get_keywords(url):
        try:
            return Video.get(url)["keywords"]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def get_popular_topics(amount):
        query = VideosSearch(
            VideoSortOrder.viewCount,
            limit=amount, language='en', region='GB'
        )
        return query.result()['result']

    @staticmethod
    def get_suggestions(topic):
        suggestions = Suggestions(language='en', region='GB')
        topics = suggestions.get(topic)['result']
        del topics[0]
        return topics

    def _convert_ints(self, raw):
        fields = ['likes', 'views']
        for data in fields:
            cleaned = "".join([i for i in raw[data] if i.isdigit()])
            raw[data] = int(cleaned) if cleaned else 0
        return

    def _get_likes(self, url):
        r = requests.get(url, headers={'User-Agent': ''})
        likes = r.text[:r.text.find(' likes"')]
        return likes[likes.rfind('"') + 1:]

    def _get_views(self, url):
        r = requests.get(url, headers={'User-Agent': ''})
        views = r.text[:r.text.find(' views"')]
        return views[views.rfind('"') + 1:]

    def _no_empty_fields(self, data, expected):
        for key in data:
            if not data[key]:
                return False
        return len(data) == expected


class YouTubeSummaries(YouTubeScraper):
    def __init__(self, topic, amount):
        super().__init__()
        self.topic = topic
        self.amount = amount
        self.rate = 10
        self.videos = set()

    def execute(self):
        self._search()
        self._generate_summaries()
        self.response = self._garbage_collector()
        return self.response

    @staticmethod
    def summarise(transcript, limiter, keywords=None):
        task = "Please summarise this transcript for me in a \
        few sentences: " + transcript + "\n\nTl;dr"
        summary = summarize_yt_script_with_gpt3(task)
        summary = summary.strip(" :-")
        time.sleep(limiter)
        return summary

    def _search(self):
        query = CustomSearch(self.topic, VideoDurationFilter.short,
                             language='en', region='GB', limit=self.amount)
        while len(self.response) < self.amount:
            result = query.result()['result']
            for i in range(len(result)):
                if len(self.response) >= self.amount:
                    break
                self._insert_results(result[i])
            query.next()
        return

    def _extract_metadata(self, result):
        data = dict()
        data["keyword"] = self.topic
        data["video_id"] = result['id']
        data["video_name"] = result['title']
        data["channel_name"] = result['channel']['name']
        data['published_at'] = result['publishedTime']
        data['views'] = self._get_views(result['link'])
        data["video_tags"] = self.get_keywords(result['link'])
        data['likes'] = self._get_likes(result['link'])
        return data

    def _insert_results(self, result):
        if in_db(self.topic, result['id']):
            return
        if result['id'] in self.videos:
            return
        data = self._extract_metadata(result)
        if not self._has_transcript_available(data):
            return
        if not self._no_empty_fields(data, 9):
            return
        self._convert_ints(data)
        self.response.append(data)
        self.videos.add(result['id'])
        return

    def _has_transcript_available(self, data):
        try:
            raw = YouTubeTranscriptApi.get_transcript(
                data['video_id'], languages=['en', 'en-GB'])
            transcript = self._format_transcript(raw)
            data['transcript'] = transcript
        except (youtube_transcript_api.NoTranscriptFound,
                youtube_transcript_api.TranscriptsDisabled,
                youtube_transcript_api.NoTranscriptAvailable,
                youtube_transcript_api.YouTubeRequestFailed):
            return False
        return True

    def _format_transcript(self, raw_transcript):
        final_transcript = []
        for text in raw_transcript:
            word = text.get('text')
            if not word or word == '[Music]':
                continue
            final_transcript.append(word)
        return " ".join(final_transcript)

    def _generate_summaries(self):
        for i, data in enumerate(self.response):
            try:
                transcript = data['transcript']
                data["summary"] = self.summarise(transcript, self.rate)
            except (RateLimitError, ServiceUnavailableError,
                    InvalidRequestError):
                continue
        return

    def _garbage_collector(self):
        new_response = []
        for data in self.response:
            if "summary" in data:
                del data['transcript']
                new_response.append(data)
        return new_response


class YouTubeUpdates(YouTubeScraper):
    PREFIX = "https://www.youtube.com/watch?v="

    def __init__(self, video_id):
        self.video_id = video_id

    def execute(self):
        for id in self.video_id:
            url, data = self.PREFIX + id, {}
            data['views'] = self._get_views(url)
            data['likes'] = self._get_likes(url)
            data['video_id'] = id
            self._insert_results(data)
        return self.response

    def _insert_results(self, results):
        if not self._no_empty_fields(results, 3):
            return
        self._convert_ints(results)
        self.response.append(results)
        return


def get_most_popular_video_transcripts_by_topic(topic, amount):
    parser = YouTubeSummaries(topic, amount)
    return parser.execute()


def get_updated_likes_and_views(video_id):
    parser = YouTubeUpdates(video_id)
    return parser.execute()
