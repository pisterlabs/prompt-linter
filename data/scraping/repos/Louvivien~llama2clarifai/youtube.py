from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import YoutubeLoader
from fastapi import APIRouter
import os
import shutil
from langchain.vectorstores import FAISS
import traceback
import datetime
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

def log_error(data_id, error_message, traceback_str):
    error_folder = "train_error_logs"
    os.makedirs(error_folder, exist_ok=True)
    error_log_path = os.path.join(error_folder, f"error_log_{data_id}.txt")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_log_path, "a") as error_log:
        error_log.write(f"[{timestamp}] {error_message}\n")
        error_log.write(traceback_str + "\n")



@router.post("/traintube")
async def traintube(data: dict):
    link = data.get('link')
    data_id = data.get('data_id')
    persist_directory = f'trained_db/{data_id}/{data_id}_all_embeddings'

 
    urls = []
    urls.append(link)
    # Directory to save audio files
    
    try:        
                
        try:
        
            for url in urls:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=True,language=["en", "id"],translation="en")
                docs=loader.load()
            if len(docs) == 0:
                save_dir = f"YouTube/{data_id}"
                # Create save directory if it doesn't exist
                os.makedirs(save_dir, exist_ok=True)
                # Transcribe the videos to text
                loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
                docs = loader.load()
                shutil.rmtree(save_dir)

        
            embeddings = OpenAIEmbeddings()
            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
            split_docs = text_splitter.split_documents(docs)
            new_vectordb = FAISS.from_documents(split_docs, embeddings)
                
            try:
                old_vectordb = FAISS.load_local(persist_directory, embeddings)
                old_vectordb.merge_from(new_vectordb)
                old_vectordb.save_local(persist_directory)
                print("Previous Embeddings were loaded.")
                return {'message':" Embeddings Generated Sucessfully."}
            except:
                new_vectordb.save_local(persist_directory)
                return {'message':" New Embeddings Generated Sucessfully."}



        except Exception as e:
                # Log the traceback information to a file
                error_message = f"Video not found check logs for details: {str(e)}"
                traceback_str = traceback.format_exc()
                log_error(data_id, error_message, traceback_str)
                return {'message': "Video not found check logs for details"}

        finally:
            print("complete")
                

    except Exception as e:
        # Log the traceback information for the general error
        error_message = f"General Error: {str(e)}"
        traceback_str = traceback.format_exc()
        log_error(data_id, error_message, traceback_str)
        return {'message': 'An error occurred. Please check the logs.'}


