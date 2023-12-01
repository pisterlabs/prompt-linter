from youtube_transcript_api import YouTubeTranscriptApi
import re
import numpy as np
from pytube import YouTube
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.llms import OpenAI
import ast

# function to check if a given URL is a valid YouTube link
def is_youtube_url(url):
    # regex pattern to match YouTube video URLs
    pattern = r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+(&\S*)?|https?://youtu\.be/[\w-]+'
    return re.match(pattern, url) is not None

# function to extract the video ID from a YouTube video URL
def get_youtube_id(url):
    # parse the video ID from the URL using the pytube module
    yt = YouTube(url)
    return yt.video_id if yt else None

def youtube_transcript(video_id):
    response = YouTubeTranscriptApi.get_transcript(video_id)
    # transcript = ""
    # for sentence in response:
    #     transcript += sentence['text'] + " "
    return response

# Calculate the total duration of the transcript.
def get_youtube_chunks(transcript, chunk_size=100):
    total_duration = transcript[-1]['start'] + transcript[-1]['duration']

    # Calculate the duration of each 100-word chunk.
    words_per_chunk = chunk_size
    # Split the transcript into 100-word chunks.
    chunks = []
    chunk = {'text': '', 'start': 0}
    for line in transcript:
        chunk['text'] += line['text'] + ' '
        if len(chunk['text'].split()) >= words_per_chunk:
            chunk['end'] = line['start'] + line['duration']
            chunks.append(chunk)
            chunk = {'text': '', 'start': line['start'] + line['duration']}
    if chunk['text']:
        chunk['end'] = total_duration
        chunks.append(chunk)

    # Print each chunk with its start and end timing.
    original_docs = []
    for chunk in chunks:
        original_docs.append({"text":chunk["text"] , "start":chunk["start"] , "end":chunk["end"]})
    return original_docs


def store_faiss_vectors(original_docs):
    docs = []
    metadatas = []
    for i, d in enumerate(original_docs):
        metadatas.extend([{"source":{"start":d["start"], "end": d["end"]}}])
        docs.append(d["text"])
    # Here we create a vector store from the documents and save it to disk.
    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    # faiss.write_index(store.index, "docs.index")
    return store

def read_faiss_index(store):
    # index = faiss.read_index("docs.index")
    # store.index = index
    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
    return chain

def play_video_from_start_end_end(video_id, start_time, end_time):
    start_time = int(np.floor(float(start_time)))
    end_time = int(np.ceil(float(end_time)))
    return f"https://www.youtube.com/embed/{video_id}?start={start_time}&end={end_time}&autoplay=1"

def get_output_video_url(video_id, lm_output):
    time_intervals = ast.literal_eval(f"[{lm_output['sources']}]")
    first_start = time_intervals[0]['start']
    last_end = time_intervals[-1]['end']
    out_video_url = play_video_from_start_end_end(video_id, first_start, last_end)
    return out_video_url
