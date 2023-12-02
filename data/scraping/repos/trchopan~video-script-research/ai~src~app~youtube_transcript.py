from datetime import datetime
from multiprocessing.pool import ThreadPool
from fastapi import HTTPException, status
import requests
from typing import List, Tuple
from urllib.parse import parse_qs, urlparse
from peewee import CharField, DateTimeField, FloatField, IntegerField, TextField
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from app.vector_store import VectorStore
from app.helpers import find_element


from .base_model import BaseDBModel, from_int, from_str, get_db, to_float


class YoutubeVideo(BaseDBModel):
    video_id = CharField()
    thumbnail = CharField(max_length=1024)
    title = CharField()
    description = TextField()
    channel = CharField()
    channel_id = CharField()
    publish_at = DateTimeField()

    def to_dict(self):
        return {
            "video_id": self.video_id,
            "thumbnail": self.thumbnail,
            "title": self.title,
            "description": self.description,
            "channel": self.channel,
            "channel_id": self.channel_id,
            "publish_at": str(self.publish_at),
        }


class YoutubeTranscript(BaseDBModel):
    video_id = CharField()
    chunk = IntegerField()
    start = FloatField()
    text = TextField()
    learn_japanese = TextField()

    def to_dict(self):
        return {
            "video_id": self.video_id,
            "chunk": self.chunk,
            "start": self.start,
            "text": self.text,
            "learn_japanese": self.learn_japanese,
        }


class YoutubeTranscriptSimilarity:
    namespace: str
    document: str
    chunk: int
    content: str
    start: float
    similarity: float

    def __init__(
        self,
        namespace: str,
        document: str,
        chunk: int,
        content: str,
        start: float,
        similarity: float,
    ) -> None:
        self.namespace = namespace
        self.document = document
        self.chunk = chunk
        self.content = content
        self.start = start
        self.similarity = similarity

    def to_dict(self) -> dict:
        result: dict = {}
        result["namespace"] = from_str(self.namespace)
        result["document"] = from_str(self.document)
        result["chunk"] = from_int(self.chunk)
        result["content"] = from_str(self.content)
        result["start"] = to_float(self.start)
        result["similarity"] = to_float(self.similarity)
        return result


class YoutubeTranscriptService:
    _EMBEDDING_NAMESPACE = "youtube-transcript"
    _CHUNK_SIZE = 1000
    _CHUNK_OVERLAP = 200

    def __init__(
        self,
        api_key: str,
        chat35: ChatOpenAI,
        embeddings: OpenAIEmbeddings,
        vector_store: VectorStore,
    ):
        self.api_key = api_key
        self.chat35 = chat35
        self.embeddings = embeddings
        self.vector_store = vector_store

    _puncturation_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "I want you to format and put in correct punctuations for the below text."
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )

    def delete_youtube_video_id(self, video_id: str):
        YoutubeVideo.delete().where(YoutubeVideo.video_id == video_id).execute()
        YoutubeTranscript.delete().where(YoutubeTranscript.video_id == video_id).execute()

    def get_videos(self):
        videos: list[YoutubeVideo] = list(YoutubeVideo.select().order_by(YoutubeVideo.publish_at))
        return videos

    def get_video(self, video_id: str) -> YoutubeVideo:
        obj: YoutubeVideo = YoutubeVideo.get(YoutubeVideo.video_id == video_id)
        if obj is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        return obj

    def pull_video_details(self, video_id: str) -> YoutubeVideo:
        YoutubeVideo.delete().where(YoutubeVideo.video_id == video_id).execute()

        # Start processing new video

        payload = {
            "id": video_id,
            "part": "contentDetails,snippet",
            "key": self.api_key,
        }
        response = requests.get("https://www.googleapis.com/youtube/v3/videos", params=payload)
        if response.status_code != 200:
            print(response.text)
            raise Exception("error getting youtube details")

        resp_dict = response.json()
        snippet = resp_dict["items"][0]["snippet"]
        publish_at = datetime.strptime(snippet["publishedAt"], "%Y-%m-%dT%H:%M:%S%z")
        yt_video = YoutubeVideo(
            video_id=video_id,
            thumbnail=snippet["thumbnails"]["standard"]["url"],
            title=snippet["title"],
            description=snippet["description"],
            channel=snippet["channelTitle"],
            channel_id=snippet["channelId"],
            publish_at=publish_at,
        )
        yt_video.save()

        return yt_video

    def get_transcript(self, video_id: str) -> list[YoutubeTranscript]:
        transcripts: List[YoutubeTranscript] = list(
            YoutubeTranscript.select()
            .where(YoutubeTranscript.video_id == video_id)
            .order_by(YoutubeTranscript.chunk)
        )

        if transcripts is None or len(transcripts) == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        return transcripts

    def parse_transcript(self, video_id: str, language="en") -> list[YoutubeTranscript]:
        YoutubeTranscript.delete().where(YoutubeTranscript.video_id == video_id).execute()

        self.vector_store.delete_embeddings(
            namespace=self._EMBEDDING_NAMESPACE,
            document=video_id,
        )

        # Start processing new transcript

        youtube_transcripts_: list[dict] = YouTubeTranscriptApi.get_transcript(
            video_id, languages=(language,)
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=self._CHUNK_OVERLAP,
        )

        # The following is a bit complicated
        # I'm trying to find out the chunk and the `start` time of that chunk by
        # checking if the transcript_has_text. It use first 20 characters to confirm it.
        youtube_transcripts: list[dict] = []
        for i, t in enumerate(youtube_transcripts_):
            if i % 3 == 0:
                youtube_transcripts.append(
                    {
                        "text": t.get("text", ""),
                        "start": t.get("start"),
                    }
                )
                if len(youtube_transcripts) > 2:
                    # Add a little bit of overlap
                    youtube_transcripts[-1]["text"] = (
                        youtube_transcripts[-2]["text"][-20:] + youtube_transcripts[-1]["text"]
                    )
            else:
                youtube_transcripts[-1]["text"] += "\n" + t.get("text", "")

        transcripts_fulltext = "\n".join([t.get("text", "").strip() for t in youtube_transcripts_])
        splitted_texts = text_splitter.split_text(transcripts_fulltext)
        transcript_starts = []

        def transcript_has_text(yt_transcript: dict, splited_text: str) -> bool:
            _text_to_check = (
                yt_transcript.get("text", "").replace(" ", "").replace("\n", "").strip()
            )
            _splitted_text = splited_text.replace(" ", "").replace("\n", "").strip()

            return _splitted_text[0:20] in _text_to_check

        for text in splitted_texts:
            found = find_element(
                youtube_transcripts,
                lambda transcript: transcript_has_text(transcript, text),
            )
            transcript_starts.append(found.get("start", 0.0) if found is not None else 0.0)

        # Start processing the functuation format in a ThreadPool

        def _process_puncturation(video_id: str, index: int, total: int, text: str):
            print(f">>> processing {self._EMBEDDING_NAMESPACE} {video_id}: " f"{index + 1}/{total}")
            prompt = self._puncturation_prompt.format_prompt(text=text)
            result = self.chat35(prompt.to_messages())
            embeddings = self.embeddings.embed_query(result.content)
            return index, result.content, embeddings

        results: list[Tuple[int, str, list[float]]] = []
        with ThreadPool(processes=5) as pool:
            workers = [
                pool.apply_async(
                    _process_puncturation,
                    (video_id, i, len(splitted_texts), text),
                )
                for i, text in enumerate(splitted_texts)
            ]
            for res in workers:
                index, content, embeddings = res.get()
                results.append((index, content, embeddings))

        # Put the result transcripts into vector store

        transcripts = []
        for [(index, content, embeddings), start] in zip(results, transcript_starts):
            transcript = YoutubeTranscript(
                video_id=video_id, chunk=index, start=start, text=content, learn_japanese=""
            )
            transcript.save()

            self.vector_store.insert_embeddings(
                namespace=self._EMBEDDING_NAMESPACE,
                document=video_id,
                chunk=index,
                embeddings=embeddings,
            )

            transcripts.append(transcript)

        return transcripts

    def get_embeddings(self, video_id: str):
        embeddings = self.vector_store.get_embeddings(
            namespace=self._EMBEDDING_NAMESPACE, document=video_id
        )
        return embeddings

    def get_similarity(self, query: str, links: List[str], k: int = 5):
        video_ids = [self.get_youtube_video_id(link) or "" for link in links]
        query_embedding = self.embeddings.embed_query(query)

        similarity_results: List[YoutubeTranscriptSimilarity] = []
        single_limit = round(k / len(links))
        for video_id in video_ids:
            similarities = self.vector_store.similarity_search(
                query_embedding,
                namespace=self._EMBEDDING_NAMESPACE,
                document=video_id,
                limit=single_limit,
            )
            transcripts: List[YoutubeTranscript] = list(
                YoutubeTranscript.select().where(YoutubeTranscript.video_id == video_id)
            )
            results = []
            for sim in similarities:
                transcript = find_element(transcripts, lambda t: t.chunk == sim.chunk)
                start = transcript.start if transcript is not None else 0.0
                content = str(transcript.text) if transcript is not None else ""
                results.append(
                    YoutubeTranscriptSimilarity(
                        namespace=self._EMBEDDING_NAMESPACE,
                        document=sim.document,
                        chunk=sim.chunk,
                        start=start + 0.0,
                        content=content,
                        similarity=sim.similarity,
                    )
                )

            similarity_results.extend(results)

        similarity_results.sort(key=lambda r: r.similarity, reverse=True)
        return similarity_results[0:k]

    def get_youtube_video_id(self, link: str) -> str | None:
        """
        Examples:
        - http://youtu.be/SA2iWivDJiE
        - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
        - http://www.youtube.com/embed/SA2iWivDJiE
        - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
        """
        query = urlparse(link)
        if query.hostname == "youtu.be":
            return query.path[1:]
        if query.hostname in ("www.youtube.com", "youtube.com"):
            if query.path == "/watch":
                p = parse_qs(query.query)
                return p["v"][0]
            if query.path[:7] == "/embed/":
                return query.path.split("/")[2]
            if query.path[:3] == "/v/":
                return query.path.split("/")[2]
        # fail?
        return None


# Create table if not exists
get_db().create_tables([YoutubeVideo, YoutubeTranscript])
