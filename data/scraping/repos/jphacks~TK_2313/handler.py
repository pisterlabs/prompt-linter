import os
import openai
from dotenv import load_dotenv
from services import whisper_call, ChatGPT_call, voicevox_call
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from supabase import create_client

load_dotenv("../.env")
openai.organization = os.environ.get('OPENAI_ORG_KEY')
openai.api_key = os.environ.get('OPENAI_API_KEY')
DB_URL = os.environ.get('DB_URL')
DB_KEY = os.environ.get('DB_KEY')


db_client = create_client(DB_URL, DB_KEY)
# イベントによって呼び出される関数


def handle_voice_driven(voice_input: bytes):
    transcript = whisper_call.whisper_transcription(voice_input)
    replay = ChatGPT_call.GPT_call(transcript)
    audio_bytes = voicevox_call.vvox_test(replay)
    # return wav as bytes
    return audio_bytes, replay


def handle_anchor_driven(anchor_uuid: str):
    response = db_client.table('SpaceAnchor').select(
        "name", "action").eq('uuid', anchor_uuid).execute()
    action = response.data
    if action is None:
        return None
    transcript = whisper_call.whisper_transcription(action)
    replay = ChatGPT_call.GPT_call(transcript)
    audio_bytes = voicevox_call.vvox_test(replay)
    return audio_bytes, replay, action
