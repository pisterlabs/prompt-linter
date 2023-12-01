# https://youtu.be/sHTAYJ_HSkA
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url="https://youtu.be/sHTAYJ_HSkA"
save_dir="examples/docs/youtube"
loader = GenericLoader(YoutubeAudioLoader([url], save_dir),OpenAIWhisperParser())
docs = loader.load()

docs[0].page_content[0:500]

