from yt_dlp import YoutubeDL
import openai
import os
from llama_index import StorageContext, StringIterableReader, GPTVectorStoreIndex, load_index_from_storage

STORAGE_PATH = "./storage"


def get_audio_file(url: str):

    ydl_opts = {
        'outtmpl': './audio.%(ext)s',
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }],
    }

    ydl = YoutubeDL(ydl_opts)
    ydl.download([url])


def get_audio_text(file_path: str) -> str:
    audio_file = open(file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


def get_summary_text(audio_text: str, summary_prompt: str) -> str:
    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)
    try:
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_PATH)
        vector_index = load_index_from_storage(storage_context)
    except Exception:
        documents = StringIterableReader().load_data(texts=[audio_text])
        vector_index = GPTVectorStoreIndex.from_documents(documents)
        vector_index.storage_context.persist(persist_dir=STORAGE_PATH)

    query_engine = vector_index.as_query_engine()
    response = query_engine.query(summary_prompt)

    return response.response
