import os
import sys

import time
import pickle

from pytube import YouTube
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
# # ==



yt = YouTube("https://www.youtube.com/watch?v=pCvxutpn8g8")
yt.streams.filter(only_audio=True).first().download \
    (output_path="/Users/a11/PycharmProjects/OSS_Project/data", filename="test.mp3")

start = time.time()
model = whisper.load_model("small")
result = model.transcribe("/Users/a11/PycharmProjects/OSS_Project/data/test.mp3")
end = time.time()

print(result["text"])
print(f"{end -start:.2f}sec")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=50,
    length_function=len,)

docs = [Document(page_content=x) for x in text_splitter.split_text(result["text"])]

split_docs = text_splitter.split_documents(docs)

with open("/Users/a11/PycharmProjects/OSS_Project/data/split_example_small.pkl", "wb") as f:
    pickle.dump(split_docs, f)

# base 119.56sec
# small 268.38sec