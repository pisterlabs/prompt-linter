import os

from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from shared_funcs import text_splitter
from langchain.embeddings.openai import OpenAIEmbeddings


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]


url = "https://www.youtube.com/watch?v=Go6YXdIWRFs&ab_channel=TheKneesovertoesguy"
loader = YoutubeLoader.from_youtube_url(url, 
                                        add_video_info=True, 
                                        translation="en",
                                        )
embeddings = OpenAIEmbeddings()
# TODO: split videos in case of long videos
# TODO: add all yt videos programmatically
# first add to other index

import pinecone

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)


index_name = "kneesovertoesguy"

result = loader.load()
for video in result:
    video.metadata['url'] = url
print (type(result))
print (f"Found video from {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long")
print ("")
print (result)
pass