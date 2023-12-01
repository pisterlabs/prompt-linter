import openai
from dotenv import load_dotenv
import os
import aiohttp
import io
from aiohttp import FormData
import json
import ffmpeg
import tempfile
import os

load_dotenv()

# openai.api_type = os.getenv("OPENAI_API_TYPE")
# openai.api_version = os.getenv("OPENAI_API_VERSION")
# openai.api_base = os.getenv("OPENAI_API_BASE")
# openai.api_key = os.getenv("OPENAI_API_KEY")

# async def transcribe(content):        
#     headers = {        
#         "api-key": f"{openai.api_key}"
#     }
#     deployment_name = "iaff_whisper"
#     whisper_version = "2023-09-01-preview"
#     deployment_api = f"{openai.api_base}openai/deployments/{deployment_name}/audio/transcriptions?api-version={whisper_version}"

#     audio_io = io.BytesIO(content)

#     data = FormData()
#     data.add_field('file',
#                    audio_io,
#                    filename="test.wav",
#                    content_type='audio/wav') 

#     transcription = ""
#     async with aiohttp.ClientSession() as session:
#         async with session.post(deployment_api, headers=headers, data=data) as response:
#             if response.status == 200:
#                 transcription = await response.json()
#             else:
#                 error_details = await response.text()
#                 print(f"Error: {response.status}, Details: {error_details}")
    
#     return transcription


def convert_webm_bytes_to_wav_bytes(webm_bytes):
    webm_fd, webm_path = tempfile.mkstemp(suffix='.webm')
    with os.fdopen(webm_fd, 'wb') as temp_webm:
        temp_webm.write(webm_bytes)

    wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(wav_fd)

    try:
        ffmpeg.input(webm_path).output(wav_path, format='wav', y=None).run()

        with open(wav_path, 'rb') as temp_wav:
            wav_bytes = temp_wav.read()
            return wav_bytes
    except ffmpeg.Error as e:
        print("Error:", e.stderr.decode())
        return None
    finally:
        os.remove(webm_path)
        os.remove(wav_path)

async def transcribe(content, lang):
    content = convert_webm_bytes_to_wav_bytes(content)
    lang_map = {"English": "en-US",
                "Vietnamese": "vi-VN",
                "Belarusian": "be-BY",
                "Ukrainian": "uk-UA",
                "Polish": "pl-PL",
                "Russian": "ru-RU"}

    azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
    azure_region = os.getenv("AZURE_REGION")
    azure_endpoint = f"https://{azure_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language={lang_map[lang]}"
    headers = {
        "Ocp-Apim-Subscription-Key": azure_speech_key,
        "Content-Type": "audio/wav"
    }

    audio_io = io.BytesIO(content)
    
    transcription = ""
    async with aiohttp.ClientSession() as session:
        async with session.post(azure_endpoint, headers=headers, data=audio_io) as response:
            if response.status == 200:
                result = await response.json()
                transcription = result.get('DisplayText', '')
            else:
                error_details = await response.text()
                print(f"Error: {response.status}, Details: {error_details}")
    return transcription