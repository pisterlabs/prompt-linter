# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:21:38 2023

@author: marca
"""


import docx
import pandas as pd
import chardet
from pdfminer.high_level import extract_text
import os
import tiktoken
from openai_pinecone_tools import generate_response, transcribe_using_whisper, ELEVENLABS_API_KEY
import pytesseract
from PIL import Image, ImageFile
import pyaudio
import wave
from elevenlabs import generate, play, voices
from elevenlabs import set_api_key

set_api_key(ELEVENLABS_API_KEY)


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Set ImageFile to accept truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Declare tesseract.exe
tesseract_path = "C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = tesseract_path


def ocr_read_image(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        raise ValueError("The file does not exist.")

    # Check if the file has a supported image extension
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext not in supported_extensions:
        raise ValueError(
            "Unsupported file extension. Please provide a supported image file."
        )

    # Open the image file
    image = Image.open(image_path)

    # Perform OCR on the image
    text = pytesseract.image_to_string(image)

    return text


def read_pdf(file_path):
    try:
        text = extract_text(file_path)
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {str(e)}")
        text = ""
    return text


def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = " ".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {str(e)}")
        text = ""
    return text


def read_xlsx_file(file_path):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading XLSX file {file_path}: {str(e)}")
        df = None

    file_name = os.path.basename(file_path)
    return file_name, df


def read_csv_file(file_path):
    try:
        with open(file_path, "rb") as f:
            encoding = chardet.detect(f.read())["encoding"]
        df = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {str(e)}")
        df = None

    file_name = os.path.basename(file_path)
    return file_name, df


def read_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading TXT file {file_path}: {str(e)}")
        text = ""
    return text


def csv_id_agent(context):
    if len(context) > 2000:
        context = context[:2000]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "You are my CSV Indentification Assistant. Your job is to take a block of text provided below, and decide if the text represents all or part of a comma separated value text.  You must decide, and must answer using either 'yes' or 'no'.",
        },
        {"role": "user", "content": f"Text to be identified: {context}"},
    ]

    response = generate_response(
        messages, temperature=0.0, n=1, max_tokens=10, frequency_penalty=0
    )
    is_csv = None

    if "yes" in response.lower():
        is_csv = True
    elif "no" in response.lower():
        is_csv = False

    else:
        print("I can't tell is this is a CSV, I'm sorry!")
        return

    return is_csv


def process_table_file(file_path):
    def count_tokens(text):
        tokens = len(encoding.encode(text))
        return tokens

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".csv":
        file_name, df = read_csv_file(file_path)
    elif file_ext == ".xlsx":
        file_name, df = read_xlsx_file(file_path)
    else:
        raise ValueError(
            "Unsupported file extension. Please provide a CSV or XLSX file."
        )

    if df is None:
        return None

    plain_csv_text = df.to_csv(index=False)
    rows = plain_csv_text.split("\n")

    chunk_tokens = 0
    chunks = []
    chunk = ""

    for row in rows:
        row_tokens = len(encoding.encode(row))
        chunk_tokens += row_tokens

        if chunk_tokens >= 1000:
            chunks.append(chunk)
            chunk = ""
            chunk_tokens = row_tokens

        chunk += "\n" + row

    # Add the remaining chunk, if any
    if chunk:
        chunks.append(chunk)

    chunks = [f"{file_name}\n{chunk}" for chunk in chunks]

    return chunks


def ingester(file_path):
    extension = file_path.split(".")[-1].lower()
    if extension == "pdf":
        return read_pdf(file_path)
    elif extension in ["doc", "docx"]:
        return read_docx(file_path)
    elif extension in ["csv", "xlsx"]:
        return process_table_file(file_path)
    elif extension == "txt":
        return read_txt(file_path)
    elif extension in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        return ocr_read_image(file_path)
    else:
        print(f"Unsupported file type: {extension}")
        return ""


def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        # Check if adding the word to the current chunk would exceed the chunk size
        if len(current_chunk) + len(word) + 1 > chunk_size:
            # If so, add the current chunk to the chunks list and start a new chunk with the current word
            chunks.append(current_chunk.strip())
            current_chunk = word
        else:
            # Otherwise, add the word to the current chunk
            current_chunk += f" {word}"

    # Add the last chunk to the chunks list
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def ingest_folder(folder_path, progress=True):
    context_chunks = []

    # List all files in the folder
    file_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    total_files = len(file_paths)

    for i, file_path in enumerate(file_paths):
        if progress:
            print(f"Processing {file_path}")

        text = ingester(file_path)

        if isinstance(text, str):
            chunks = chunk_text(text)
            context_chunks.extend(chunks)

        else:
            context_chunks.extend(text)

    return context_chunks



def record_audio(filename, duration=5):
    # Set the sample format, channels, rate, and chunk size
    sample_format = pyaudio.paInt16
    channels = 1
    rate = 44100
    chunk = 1024

    # Initialize the PyAudio object
    audio = pyaudio.PyAudio()

    # Open the stream for recording
    stream = audio.open(format=sample_format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

    # Record audio
    print("Listening...")
    frames = []
    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PyAudio object
    audio.terminate()

    # Save the recorded audio as a .wav file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print("Response saved")
    



def listen(duration=5):
    # Record audio and save it as a .wav file
    audio_file = "recorded_audio.wav"
    record_audio(audio_file, duration)

    # Transcribe the recorded audio
    transcript = transcribe_using_whisper(audio_file)

    # Remove the temporary .wav file
    if os.path.exists(audio_file):
        os.remove(audio_file)
    
    print(transcript)
    return transcript



def text_to_speech(text, voice_name="Rachel"):
    # Get the list of available voices
    available_voices = voices()

    # Find the desired voice
    voice = next((v for v in available_voices if v.name == voice_name), None)

    if voice is not None:
        # Generate audio from the text
        audio = generate(text=text, voice=voice.voice_id, model="eleven_monolingual_v1")

        # Play the generated audio
        play(audio)
    else:
        print(f"Voice '{voice_name}' not found.")





