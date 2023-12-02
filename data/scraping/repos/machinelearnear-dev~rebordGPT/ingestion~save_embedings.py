import time
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from typing import List

load_dotenv()

def to_chunks(name, link, transcription_path, chunk_length=1000):
    try:
        with open(transcription_path, 'r') as f:
            transcript = json.loads(f.read())
        start = None
        chunks = []
        metadatas = []
        chunk = ""
        for item in transcript:
            if start is None:
                start = item['start']
            if(len(item['text']) > chunk_length):
                item['text'] = item['text'][:chunk_length - 100]
            temp_chunk = chunk + " " + item['text'] + " "
            if(len(temp_chunk) <= chunk_length):
                chunk = temp_chunk
            else:
                chunks.append(chunk)
                metadata = {'name': name, 'link': link, 'start': start}
                metadatas.append(metadata)
                start = item['start']
                chunk = item['text']

        return chunks, metadatas
    except Exception as e:
        print("Error reading transcription", e)
        return [], []

def save_embedings(persist_directory: str = "db", chunks: list = None, metadatas: list = None):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(chunks, embeddings, metadatas=metadatas, persist_directory=persist_directory)

def save_updated_episodes(episodes: List, filename: str = "episodes.json"):
    with open(filename, 'w') as f:
        f.write(json.dumps(episodes))

def main():
    print("Starting")
    persist_directory = "../db"
    with open('episodes.json', 'r') as f:
        episodes = json.load(f)
        if(len(episodes) == 0):
            print("No episodes to process")
            return
        for episode in episodes:
            if episode['processed'] == True or episode['transcribed'] == False:
                print("Episode already processed or not transcribed yet")
                continue
            transcription_path = episode['transcription']
            url = episode['url']
            title = episode['title']
            print("Processing", title)
            chunks, metadatas = to_chunks(title, url, transcription_path)
            if(len(chunks) == 0 or len(metadatas) == 0):
                print("No chunks to process due to an exception for", title)
                continue
            try:
                save_embedings(persist_directory, chunks, metadatas)
                episode['processed'] = True
                save_updated_episodes(episodes)
                time.sleep(1)
            except Exception as e:
                print("Error saving embedings", e)
                continue

if __name__ == "__main__":
    main()