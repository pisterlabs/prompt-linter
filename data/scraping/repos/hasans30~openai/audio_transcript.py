import os
import openai
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv() 

api_key=openai_api_key = os.getenv('OPENAI_API_KEY','no api key found');

openai.api_key = api_key



# Transcription 
def transcribe_audio(file_path, video_id):
        # The path of the transcript
        transcript_filepath = f"tmp/{video_id}.txt"

        # Get the size of the file in bytes
        file_size = os.path.getsize(file_path)

        # Convert bytes to megabytes
        file_size_in_mb = file_size / (1024 * 1024)

        # Check if the file size is less than 25 MB
        if file_size_in_mb < 25:
            with open(file_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                
                # Writing the content of transcript into a txt file
                with open(transcript_filepath, 'w') as transcript_file:
                    transcript_file.write(transcript['text'])

        else:
            print("Please provide a smaller audio file (less than 25mb).")


transcribe_audio("tmp/meetingnote.mp3", "1")