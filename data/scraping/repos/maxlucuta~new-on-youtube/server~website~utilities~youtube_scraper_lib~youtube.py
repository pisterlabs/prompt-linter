from openai.error import RateLimitError
from openai.error import ServiceUnavailableError
from openai.error import InvalidRequestError
from .scrapers.metadata_scraper import MetaDataScraper
from .scrapers.transcript_scraper import TranscriptScraper
from .cleaners.metadata_cleaner import MetaDataCleaner
from .cleaners.transcript_cleaner import TranscriptCleaner
from .cleaners.data_cleaner import DataCleaner as dc
from .proxy.proxies import Proxy
from ..gpt3 import summarize_yt_script_with_gpt3
from ..database import db_contains_video


class YouTubeScraperFactory:
    """Project specific factory class that abstracts library class methods
       for project specific YouTube scraping."""

    def __init__(self, metadata_scraper: object, transcript_scraper:
                 object, amount: int, proxy_service: object):
        """Contructs YouTubeScraperFactory object.

        Args:
            metadata_scraper (object): MetaDataScraper object
            transcript_scraper (object): TranscriptScraper object
            amount (int): amount of videos required
        """

        self.metadata_scraper = metadata_scraper
        self.transcript_scraper = transcript_scraper
        self.proxy_service = proxy_service
        self.metadata_cleaner = MetaDataCleaner()
        self.transcript_cleaner = TranscriptCleaner()
        self.amount = amount
        self.videos = set()
        self.result = []

    def execute(self) -> list[dict[str, any]]:
        """Returns the executed query in suitable format.

        Returns:
            list[dict[str, any]]: full response for a scraping query
            in the format required for database insertions.
        """

        self._scrape()
        self.transcript_cleaner.full_clean(self.result)
        self._summarise_transcripts()
        self._remove_failed_summaries()
        return self.result

    def _scrape(self):
        """Scrapes YouTube and attempts to get retrieve unique metadata
           and transcripts, until the amount of data retrieved is
           equal to the required amount. Faulty responses are skipped.
        """

        response = self.metadata_scraper.execute()
        while len(self.result) < self.amount:
            try:
                metadata = next(response)
            except StopIteration:
                break
            self.metadata_cleaner.full_clean(metadata)
            if self._check_metadata_status(metadata):
                video_id = metadata["video_id"]
                transcript = self.transcript_scraper.execute(video_id)
                raw = transcript["transcript"]
                try:
                    if not self._check_transcript_status(raw):
                        continue
                except Exception:
                    print("Ran out of Proxies!", flush=True)
                    break
                metadata.update(transcript)
                self.result.append(metadata)
                self.videos.add(video_id)

    def _summarise_transcripts(self):
        """ Summarises all transcrips stored in self.response using
            ChatGPT API, failed summaries are discarded.
        """

        for data in self.result:
            try:
                transcript = data['transcript']
                data["summary"] = self.transcript_scraper.summarise(
                    transcript, summarize_yt_script_with_gpt3, 0)
            except (RateLimitError, ServiceUnavailableError,
                    InvalidRequestError):
                continue

    def _check_metadata_status(self, metadata: dict[str, str]) -> bool:
        """Performs necessary checks on retrieved metadata before
           comitting to transcript scraping.

        Args:
            metadata (dict[str, str]): scraped metadata

        Returns:
            bool: False if the video has been previously scraped,
            exists already in the DB, or has missing fields. True
            otherwise.
        """

        video_id = metadata["video_id"]
        topic = metadata["keyword"]
        duration = metadata["duration"]
        visible = metadata["visible"]
        if video_id in self.videos or not dc.occupied_fields(metadata, 10):
            return False
        if db_contains_video(topic, video_id) or not visible:
            return False
        if not self._check_video_duration(duration):
            return False
        return True

    def _check_transcript_status(self, response: str) -> bool:
        """Checks if a recieved transcript is valid, and if not,
           handles any issues that occured.

        Args:
            response (str): raw transcript response

        Returns:
            bool: True if a valid transcript exists, False otherwise.
            Will rotate proxies if IP gets blocked.

        Raises:
            Exception: if there are no more proxies available
        """

        if not response or response == "404":
            return False
        if response == "blocked":
            if not self._rotate_proxies():
                raise Exception
            return False
        return True

    def _rotate_proxies(self) -> bool:
        """Rotates proxies for all scrapers.

        Returns:
            bool: True if successfull, false if there are
            no more proxies available.
        """

        proxy = self.proxy_service.get()
        if not proxy:
            return False
        print(f"Proxy rotated to: {proxy}", flush=True)
        self.metadata_scraper.rotate_proxy(proxy)
        self.transcript_scraper.rotate_proxy(proxy)
        return True

    def _check_video_duration(self, duration: str) -> bool:
        """Checks if the duration of the video is between 1 min
           and 7 min.

        Args:
            duration (str): Duration of the video in hh:mm:ss

        Returns:
            bool: true if the video is within range, false if it
            exceeds the range
        """

        time = duration.split(":")
        if len(time) > 2:
            return False
        return (int(time[-1]) + int(time[-2]) * 60) > 60

    def _remove_failed_summaries(self):
        """Removes entries in self.result that do not have a valid
           summary, which could occur if the GPTAPI throws an error.
        """

        response = []
        for data in self.result:
            if "summary" in data:
                del data['transcript']
                del data['duration']
                response.append(data)
        self.result = response


def get_most_popular_video_transcripts_by_topic(
        topic: str, amount: int) -> list[dict[str, any]]:
    """Uses YouTubeScraperFactory class to generate metadata and summaries
       for a number of YouTube videos for a given topic, abstracts the class
       for a cleaner API.

    Args:
        topic (str): topic to be queried
        amount (int): number of expected responses

    Returns:
        list[dict[str, any]]: query response in format suitable for
        database insertion.
    """

    params = ["keyword",
              "video_id",
              "channel_name",
              "video_name",
              "published_at",
              "views",
              "likes",
              "video_tags",
              "duration",
              "visible"]

    proxy_service = Proxy(["GB"], rand=True, website="https://youtube.com")
    proxy = proxy_service.get()
    print(f"Running scraper service on proxy: {proxy}", flush=True)
    meta_scraper = MetaDataScraper(topic, params, proxy=proxy)
    transcript_scraper = TranscriptScraper(proxy=proxy)
    interface = YouTubeScraperFactory(
        meta_scraper, transcript_scraper, amount, proxy_service)
    return interface.execute()


def get_updated_metadata_by_id(video_id: str) -> dict[str, str]:
    """Retrives up to date metadata for a given video id, abstracts
       library methods to allow for a simple API.

    Args:
        video_id (str): YouTube video id

    Returns:
        dict[str, str]: dict with fields views, likes and upload
        date, with associated values in suitable DB format. Returns
        None if scraping has failed
    """

    proxy_service = Proxy(["GB"], rand=True, website="https://youtube.com")
    proxy = proxy_service.get()
    print(f"Running update service on proxy: {proxy}", flush=True)
    url = "https://www.youtube.com/watch?v=" + video_id
    views = MetaDataScraper.get_views(url)
    likes = MetaDataScraper.get_likes(url, proxy)
    date = MetaDataScraper.get_upload_date(url)
    data = {'views': views, 'likes': likes, 'published_at': date}
    MetaDataCleaner().full_clean(data)
    if not dc.occupied_fields(data, 3):
        return None
    return data
