from fastapi import APIRouter,  UploadFile
from app.config import settings
from transformers import pipeline
from openai import OpenAI
from faster_whisper import WhisperModel
import tempfile
import structlog
import os
import torch




logger = structlog.get_logger()

audio_router = r = APIRouter()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = WhisperModel('medium', device="cpu", compute_type="int8")


pipe = pipeline("automatic-speech-recognition",
                settings.whisper_model,
                generate_kwargs={"task": "transcribe", "language": "en"},
                torch_dtype=torch.float32,
                device=device)

pipe.model = pipe.model.to_bettertransformer()

    
@r.post("/transcribe")
async def transcribe_audio_local_approach1(file: UploadFile):
    """Transcribe the audio file locally using fast-whisper library

    Args:
        file (UploadFile): Audio file

    Returns:
        json: Transcript of the audio file
    """
    try:
        # Create a directory to store audio files if not exists
        os.makedirs("audio_files", exist_ok=True)

        # Save the audio file to a unique filename
        audio_path = f"audio_files/{'test'}.wav"
        
        logger.info(f"{audio_path}")

        content = await file.read()
        # Create a temporary file to store the PDF content
        with tempfile.NamedTemporaryFile(dir="audio_files", suffix=".wav", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
            
            logger.info(temp_file_path)
            
            segments, info = model.transcribe(temp_file_path, beam_size=5)

            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            segments = list(segments) 
            transcript = ""
            for segment in segments:
                transcript += segment.text
            logger.info(f"transcript: {transcript}")
            return {"transcript": transcript,"message": "Audio file received and saved successfully."}

    except Exception as e:
        # Handle errors and return an appropriate response
        return {"message": f"Error: {str(e)}"}


@r.post("/transcribe/UPDATEPATH")  # UPDATE THE PATH
async def transcribe_audio_local_approach2(file: UploadFile):
    """Transcribe the audio file using insanely-fast-whisper library

    Args:
        file (UploadFile): audio file

    Returns:
        json: transcript of the file
    """
    try:
        # Create a directory to store audio files if not exists
        os.makedirs("audio_files", exist_ok=True)

        # Save the audio file to a unique filename
        audio_path = f"audio_files/{'test'}.wav"
        
        logger.info(f"{audio_path}")
        
        
        content = await file.read()
        # Create a temporary file to store the PDF content
        with tempfile.NamedTemporaryFile(dir="audio_files", suffix=".wav", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
            
            logger.info(temp_file_path)
            
            transcript = pipe(temp_file_path,
               chunk_length_s=30,
               batch_size=24,
               language='fa',
               return_timestamps=True)

            logger.info(f"transcript: {transcript}")
            return {"transcript": transcript["text"],"message": "Audio file received and saved successfully."}

    except Exception as e:
        # Handle errors and return an appropriate response
        return {"message": f"Error: {str(e)}"}


