# HOW TO LOAD DATA FROM DIFFERENT SOURCE
# THE RESULT WILL BE ALWAYS A LIST OF DOCUMENT OBJECTS

# EACH DOCUMENT OBJECT HAS THE FOLLOWING PROPERTIES
# page_content: str
# metadata: dict
# page_number: int
# source: str

import os, sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from llms.llm import azure_openai_embeddings


# PDF
def load_pdf():
    # load a pdf using the document loader pypdf loader
    from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader("./documents/react paper.pdf")
    pages = loader.load()
    print(len(pages))
    # navigate page properties
    page = pages[0]
    print(page.page_content[:100])
    print(page.metadata)  # source and page number


# YOUTUBE
# ISSUE WITH Whisper QUOTA LIMIT
def load_youtube():
    from langchain.document_loaders.generic import GenericLoader

    # speech to text module
    from langchain.document_loaders.parsers import OpenAIWhisperParser

    # load audio file from youtube video
    from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

    from pydub import AudioSegment
    import whisper
    from pydub.silence import split_on_silence

    url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"

    save_dir = "docs/youtube/"
    loader = GenericLoader(YoutubeAudioLoader([url], save_dir), OpenAIWhisperParser())
    docs = loader.load()

    print(docs[0])


# WEB URL
def load_weburl():
    from langchain.document_loaders import WebBaseLoader

    loader = WebBaseLoader(
        "https://github.com/basecamp/handbook/blob/master/37signals-is-you.md"
    )
    docs = loader.load()

    print(docs[0].page_content[:500])


def load_notion():
    from langchain.document_loaders import NotionDirectoryLoader

    loader = NotionDirectoryLoader("TODO local notion path")

    docs = loader.load()

    print(docs[0].page_content[:500])
