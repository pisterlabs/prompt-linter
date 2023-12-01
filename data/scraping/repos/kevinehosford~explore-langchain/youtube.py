from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url = 'https://www.youtube.com/watch?v=dWqNgzZwVJQ'
save_dir="docs/youtube/"

loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir=save_dir),
    OpenAIWhisperParser()
)

docs = loader.load()
print(docs[0].page_content[0:500])