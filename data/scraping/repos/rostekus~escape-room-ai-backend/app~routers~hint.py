import io
import uuid
from os.path import join
from google.cloud import texttospeech
import openai
import soundfile as sf
from fastapi import APIRouter
import os
from app.models.hint import ChatResponse
from app.utils.serializers.audio_serializers import AudioGoogleSerializer
from app.utils.tools.generate_link import generate_download_signed_url_v4
from app.utils.tts.google_tts import Text2SpeachGoogleApiModel
from app.core.hints import riddle_hints, riddle_places
from app.core.actors import template, description, actor1


router = APIRouter()


def get_prev_hints(riddle_id, hint_id):
    prev_hints = ""
    for i in range(1, hint_id):
        prev_hints += f"{i} : "
        prev_hints += f"{riddle_hints[riddle_id,i]}\n"

    return prev_hints


text_to_speech = Text2SpeachGoogleApiModel()
text_to_speech.load()


def generate_prompt(riddle_place, prev_hint, hint, diff, comms):
    return template.format(actor1, diff, description,
        riddle_place.capitalize(), prev_hint.capitalize(), comms.capitalize(), hint.capitalize()
    )


@router.get("/api/v1/hint/{riddle_id}/{hint_id}")
async def read_user_item(hint_id: int, riddle_id: int, comms: str, difficulty:str, voice:str) -> ChatResponse:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    print(comms, difficulty, voice)
    # print((get_prev_hints(riddle_id,hint_id)))
    # print(riddle_hints[(riddle_id, hint_id)])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": generate_prompt(
                    riddle_places[riddle_id],
                    get_prev_hints(riddle_id, hint_id),
                    riddle_hints[(riddle_id, hint_id)],
                    difficulty, comms
                ),
            },
            {
                "role": "user",
                "content": " Write your hint, add a little bit of mistery and fun.",
            },
        ],
    )
    print(generate_prompt(
                    riddle_places[riddle_id],
                    get_prev_hints(riddle_id, hint_id),
                    riddle_hints[(riddle_id, hint_id)],
                    difficulty, comms
                ))

    hint_text = response["choices"][0]["message"]["content"]
    if voice ==  "Female":
        voice_kwargs = {"ssml_gender":  texttospeech.SsmlVoiceGender.FEMALE,"name": "en-US-Wavenet-F"}
    else:
        voice_kwargs = {}
    audio_buffer = text_to_speech.process(hint_text, voice_kwargs, {})
    audio, audio_sample_rate = sf.read(io.BytesIO(audio_buffer))
    serializer = AudioGoogleSerializer(audio, audio_sample_rate, "wav", "/tmp")
    audio_filename = join("audio", str(uuid.uuid4()))
    with serializer:
        serializer.serialize(
            "audio-escape-room",
            audio_filename,
        )
    url = generate_download_signed_url_v4("audio-escape-room", audio_filename + ".wav")

    return ChatResponse(
        audioUrl=url, text=riddle_hints[(riddle_id, hint_id)], aiText=hint_text
    )
