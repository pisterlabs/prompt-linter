import os
import openai
import sys
sys.path.append('../..')

import constants

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# The course will show the pip installs you would need to install packages on your own machine.
# These packages are already installed on this platform and should not be run again.
#! pip install pypdf 
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs\LLXZ.pdf")
pages = loader.load()
# len(pages) 160
page = pages[0]
# print(page.page_content[0:500])
page.metadata
# {'source': 'docs\LLXZ.pdf', 'page': 0}

#YouTube
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

# ! pip install yt_dlp
# ! pip install pydub

url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()
docs[0].page_content[0:500]

#URLs
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")
docs = loader.load()
print(docs[0].page_content[:500])

#Notion
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
print(docs[0].page_content[0:200])
docs[0].metadata