from langchain.document_loaders import YoutubeLoader
from langchain.docstore.document import Document
from typing import Any, Dict, List, Optional, Union

class SherherdYoutubeLoader(YoutubeLoader):
     def load(self) -> List[Document]:
        """Load documents."""
        try:
            from youtube_transcript_api import (
                NoTranscriptFound,
                TranscriptsDisabled,
                YouTubeTranscriptApi,
            )
        except ImportError:
            raise ImportError(
                "Could not import youtube_transcript_api python package. "
                "Please install it with `pip install youtube-transcript-api`."
            )


        if self.add_video_info:
            # Get more video meta info
            # Such as title, description, thumbnail url, publish_date
            video_info = self._get_video_info()
            title = video_info["title"]
            id = self.video_id
            print(title)
            metadata = {"source": f"Youtube - {title} - {id}"}

            # metadata.update(video_info)
            print(metadata)

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
        except TranscriptsDisabled:
            return []

        try:
            transcript = transcript_list.find_transcript(self.language)
        except NoTranscriptFound:
            en_transcript = transcript_list.find_transcript(["en"])
            transcript = en_transcript.translate(self.translation)

        transcript_pieces = transcript.fetch()

        transcript = " ".join([t["text"].strip(" ") for t in transcript_pieces])

        return [Document(page_content=transcript, metadata=metadata)]