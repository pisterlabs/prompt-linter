# A Q&A chatbot for a YouTube video in Python using Whisper and LangChain.
# Author: Huzbi
# Creation Date: 30/06/2023

import os
import openai
import faiss
import tempfile

from moviepy.editor import *
from pytube import YouTube
from urllib.parse import urlparse, parse_qs

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

os.environ["OPENAI_API_KEY"] = ""

def transcribe_audio(file_path):
    file_size = os.path.getsize(file_path)
    file_size_in_mb = file_size / (1024*1024)
    if file_size_in_mb < 25:
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        return transcript
    else:
        print("Error: File size limit exceed. Provide an audio file within 25mb.")

def divide_segment():
    return

def main():
    url = input("Enter YouTube Video URL: ")

    query = urlparse(url).query
    params = parse_qs(query)
    video_id = params["v"][0]

    with tempfile.TemporaryDirectory() as temp_dir:
        yt = YouTube(url)

        # Download the audio stream from the YouTube video's URL
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_stream.download(output_path=temp_dir)

        # Convert the audio stream from the temp_dir directory into MP3
        audio_path = os.path.join(temp_dir, audio_stream.default_filename)
        audio_clip = AudioFileClip(audio_path)
        audio_clip.write_audiofile(os.path.join(temp_dir, f"{video_id}.mp3"))

        # Save the path of the audio file for future use
        audio_path = f"{temp_dir}/{video_id}.mp3"

        # Transcribe the audio
        transcript = transcribe_audio(audio_path)

        os.remove(audio_path)

        print(transcript.text)

    # Split the text into smaller chunk sizes
    textsplitter = CharacterTextSplitter(chunk_size = 512, chunk_overlap = 0)
    texts = textsplitter.split_text(transcript.text)

    # Create a FAISS index
    store = FAISS.from_texts(
        texts, OpenAIEmbeddings(), metadatas=[{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))]
    )

    # Saving the FAISS index for future use, if there is use of it
    faiss.write_index(store.index, "doc.faiss")

    # Intiialize the OpenAI language model and configure the chain
    llm = OpenAI(temperature=0)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=store.as_retriever()
    )

    # The Q&A block of code which is run after previous steps are successfully executed
    while True:
        question = input("Question: ")
        answer = chain({"question": question}, return_only_outputs=True)
        print("Answer: ", answer["answer"])
        print("Sources: ", answer["sources"])
        print("\n")

if __name__ == "__main__":
    main()
