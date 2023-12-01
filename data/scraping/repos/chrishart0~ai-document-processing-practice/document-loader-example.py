# Video Source: https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/2/document-loading
# This file contains several examples of how you can load various types of data into langchain, such as pdf and youtbe videos.
import os
import openai
from langchain.document_loaders import PyPDFLoader

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader


openai.api_key  = os.environ['OPENAI_API_KEY']


#################
# Load from PDF #
#################
# ! pip install yt_dlp

# Load in the PDF
loader = PyPDFLoader("data/MachineLearning-Lecture01.pdf")
pages = loader.load()

print(len(pages))

page = pages[0]
print(page.page_content[0:500])

print(page.metadata)


#####################
# Load from YouTube #
#####################
# ! pip install yt_dlp
# ! pip install pydub

url="https://www.youtube.com/watch?v=0xR36cpU1EM"
save_dir="data/youtube/"

#NOTE: This uses the OpenAI Whisper model to transibe the video into text.
# loader = GenericLoader(
#     YoutubeAudioLoader([url],save_dir),
#     OpenAIWhisperParser()
# )
# docs = loader.load()
# docs[0].page_content[0:500]

#################
# Load from URL #
#################
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")
docs = loader.load()
print(docs[0].page_content[:500])