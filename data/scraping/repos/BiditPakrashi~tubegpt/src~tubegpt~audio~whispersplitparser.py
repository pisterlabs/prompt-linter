from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.schema import Document
from typing import Iterator, Optional
from langchain.document_loaders.blob_loaders import Blob

class WhisperSplitParser(OpenAIWhisperParser):
    """Transcribe and parse audio files.
    Audio transcription is with OpenAI Whisper model."""

    def __init__(self, api_key: Optional[str] = None,chunk_duration=10):
        self.api_key = api_key
        self.chunk_duration = chunk_duration # in  seconds

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        import io

        try:
            import openai
        except ImportError:
            raise ValueError(
                "openai package not found, please install it with "
                "`pip install openai`"
            )
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ValueError(
                "pydub package not found, please install it with " "`pip install pydub`"
            )

        # Set the API key if provided
        if self.api_key:
            openai.api_key = self.api_key

        # Audio file from disk
        audio = AudioSegment.from_file(blob.path)

        # Chunk duration is set to 10 secs for the purpose of time correlation
        chunk_duration_ms = self.chunk_duration * 1000

        # Split the audio into chunk_duration_ms chunks
        for split_number, i in enumerate(range(0, len(audio), chunk_duration_ms)):
            # Audio chunk
            chunk = audio[i : i + chunk_duration_ms]
            file_obj = io.BytesIO(chunk.export(format="mp3").read())
            if blob.source is not None:
                file_obj.name = blob.source + f"_part_{split_number}.mp3"
            else:
                file_obj.name = f"part_{split_number}.mp3"

            # Transcribe
            print(f"Transcribing part {split_number+1}!")
            #transcript = "hello"#openai.Audio.transcribe("whisper-1", file_obj)
            transcript = openai.Audio.transcribe("whisper-1", file_obj)

            yield Document(
                page_content=transcript.text,
                #page_content=transcript,
                metadata={"source": blob.source, "chunk": split_number, "from": i, "to":i+chunk_duration_ms},
            )
