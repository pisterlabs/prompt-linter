#!/usr/bin/env python
# encoding: utf8

import os
from google.cloud import speech, texttospeech as tts
from openai_interface import OpenAIInterface

# cred_file_incare_dialog = os.environ['GOOGLE_CONVERSATIONAL_DIALOGFLOW']

# # Instantiates a client
# client = speech.SpeechClient.from_service_account_file(cred_file_incare_dialog)

# # The name of the audio file to transcribe
# audio_file_path = '/home/nkvch/tiago_public_ws/src/rcprg/dialogflow/data/container/sound_186868.wav'
# voice_name = "pl-PL-Wavenet-B"

# def text_to_wav(voice_name, text):
#     language_code = "-".join(voice_name.split("-")[:2])
#     text_input = tts.types.SynthesisInput(text=text)
#     voice_params = tts.types.VoiceSelectionParams(
#         language_code=language_code, name=voice_name
#     )
#     audio_config = tts.types.AudioConfig(audio_encoding=tts.enums.AudioEncoding.LINEAR16)

#     client = tts.TextToSpeechClient.from_service_account_file(cred_file_incare_dialog)
#     response = client.synthesize_speech(
#         text_input, voice_params, audio_config
#     )

#     filename = "output.wav"
#     with open(filename, "wb") as out:
#         out.write(response.audio_content)
#         print "Generated speech saved to {}".format(filename)

# text_to_wav(voice_name, "jedź do kuchni")

import jsonschema

# Your updated schema
response_schema_during_task = {
    "type": "object",
    "properties": {
        "name": {
            # "type": ["string", "null"],  # Allow string or null
            "enum": ["fuz", "bar", None]  # Allow "null" or None
        },
        "unexpected_question": {"type": "boolean"},
    },
    "required": ["name", "unexpected_question"],
    "additionalProperties": False
}

# Your JSON response with "name" as null (using the string "null")
response_null = {
    "name": "null",
    "unexpected_question": True
}

# Your JSON response with "name" as None (Python None object)
response_none = {
    "name": None,
    "unexpected_question": True
}

# Validate the responses against the schema
try:
    jsonschema.validate(response_null, response_schema_during_task)
    print("Validation successful. The response with null 'name' is valid.")
except jsonschema.exceptions.ValidationError as e:
    print("Validation failed. The response with null 'name' is not valid.")
    print(e.message)

try:
    jsonschema.validate(response_none, response_schema_during_task)
    print("Validation successful. The response with 'None' 'name' is valid.")
except jsonschema.exceptions.ValidationError as e:
    print("Validation failed. The response with 'None' 'name' is not valid.")
    print(e.message)
