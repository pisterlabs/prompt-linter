import sys
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.indexes import VectorstoreIndexCreator

load_dotenv('.env')

video_id = sys.argv[1]

loader = YoutubeLoader(video_id)
docs = loader.load()

index = VectorstoreIndexCreator()
index = index.from_documents(docs)

response = index.query("Summarise the video in 3 bullet points")
print(f"Answer: {response}")