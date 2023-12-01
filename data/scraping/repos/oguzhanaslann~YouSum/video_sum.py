from youtube_transcript_api import YouTubeTranscriptApi
import os
import urllib.parse as urlparse
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv() 
embeddings = OpenAIEmbeddings()
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

def get_video_id(url):
    query = urlparse.urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = urlparse.parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    # fail?
    return None

def get_combined_text(url):
    video_id = get_video_id(url)
    response = YouTubeTranscriptApi.get_transcript(video_id)
    json_object = json.dumps(response, indent = 4)
    print(json_object)
    
    combined = ""
    for item in response:
        combined += item['text'] + "-"
    return combined

def get_qa_chain(url):
    combined = get_combined_text(url)
    texts = text_splitter.split_text(combined)
    db = Chroma.from_texts(texts, embeddings)
    qa_chain = RetrievalQA.from_chain_type(llm,retriever= db.as_retriever())
    return qa_chain
