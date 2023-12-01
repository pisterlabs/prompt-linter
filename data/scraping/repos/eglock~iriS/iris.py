import struct
import subprocess
import time
import logging
import traceback
import json
import requests
from datetime import datetime
from contextlib import contextmanager

import openai
import pvporcupine
import pvcobra
import pyaudio
from pydub import AudioSegment
from halo import Halo

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

spinner = Halo(spinner='dots')

CONFIG = {
    'chunk_size': 512,
    'sample_rate': 16000,
    'silence_threshold': 1.5,  # seconds
    'wake_word_paths': ['wake/hey_iris.ppn'],
    'format': pyaudio.paInt16,
    'channels': 1,
    'alert_sounds': {
        'wake': '/System/Library/Sounds/Ping.aiff',
        'success': '/System/Library/Sounds/Glass.aiff',
        'fail': '/System/Library/Sounds/Basso.aiff'
    },
    'hass_address': '10.10.233.190',
    'hass_port': '8123'
}

def init_picovoice_modules(access_key):
    """Initialize Porcupine and Cobra."""
    try:
        porcupine = pvporcupine.create(access_key=access_key, keyword_paths=CONFIG['wake_word_paths'])
        cobra = pvcobra.create(access_key=access_key)
        return porcupine, cobra
    except Exception as e:
        logger.error("Failed to initialize Picovoice modules: ")
        logger.error(traceback.format_exc())
        raise

@contextmanager
def audio_stream(p):
    """Context manager for PyAudio to ensure resources are cleaned up."""
    stream = p.open(
        format=CONFIG['format'],
        channels=CONFIG['channels'],
        rate=CONFIG['sample_rate'],
        input=True,
        frames_per_buffer=CONFIG['chunk_size']
    )
    try:
        yield stream
    finally:
        stream.stop_stream()
        stream.close()

def get_next_audio_frame(stream):
    """Get the next audio frame from the stream, handle overflow."""
    frame_data = stream.read(CONFIG['chunk_size'], exception_on_overflow=False)
    return struct.unpack_from("h" * CONFIG['chunk_size'], frame_data)

def save_frames_as_mp3(frames, file_path="output.mp3"):
    """Save the audio frames as an MP3 file."""
    # Convert the list of frames into a single bytes object
    frames_bytes = b''.join(frames)

    # Calculate the number of bytes per frame: 2 bytes/sample * 1 channel
    bytes_per_frame = 2 * 1  # Modify as needed based on audio format

    # Calculate if there are any leftover bytes and trim the frames_bytes if necessary
    remainder = len(frames_bytes) % bytes_per_frame
    if remainder != 0:
        logger.warning("Found an incomplete frame, trimming %s bytes.", remainder)
        frames_bytes = frames_bytes[:-remainder]

    # frames_bytes should have a length that is a multiple of bytes_per_frame
    try:
        audio_segment = AudioSegment(
            data=frames_bytes,
            sample_width=2,  # 2 bytes (16 bit)
            frame_rate=CONFIG['sample_rate'],
            channels=1  # mono
        )

        audio_segment.export(file_path, format="mp3")
        return True
    except Exception as e:
        logger.error("An error occurred while saving MP3: ")
        logger.error(traceback.format_exc())
        return False

def transcribe_audio(file_path="output.mp3"):
    """Transcribe the audio file using OpenAI."""
    try:
        with open(file_path, 'rb') as f:
            spinner.start('Waiting for response from Whisper')
            response = openai.Audio.transcribe("whisper-1", f)
            spinner.succeed("Response received from Whisper")
            return response.text
    except KeyboardInterrupt:
        spinner.fail("Keyboard interrupt")
        return None
    except Exception as e:
        logger.error("An error occurred while transcribing audio: ")
        logger.error(traceback.format_exc())
        return None

def get_api_request_via_completion(message):
    post_request_function = {
        "name": "post",
        "description": "Make a POST request to the Home Assistant REST API",
        "parameters": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint to which the request will be sent, excluding the /api/ prefix",
                    "example": "services/light/turn_on"
                },
                "body": {
                    "type": "string",
                    "description": "Body data to be sent with the request, never to be left empty ",
                    "example": "{\"entity_id\": \"all\"}"
                }
            },
            "required": ["method", "endpoint", "body"]
        },
        "response": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Response returned by the API",
                    "example": "{\n  \"latitude\": 0.00000000000000,\n  \"longitude\": 0.00000000000000,\n  \"elevation\": 0,\n  \"unit_system\": {\n    \"length\": \"mi\",\n    \"accumulated_precipitation\": \"in\",\n    \"mass\": \"lb\",\n    \"pressure\": \"psi\",\n    \"temperature\": \"°F\",\n    \"volume\": \"gal\",\n    \"wind_speed\": \"mph\"\n  },\n  \"location_name\": \"1600 Pennsylvania Avenue\",\n  \"time_zone\": \"America\/New_York\",\n  \"components\": [\n    \"hue\",\n    \"api\",\n    \"zone\",\n    \"button\",\n    \"fan\",\n    \"homekit\",\n    \"media_player\",\n    \"switch\",\n    \"weather\",\n    \"history\",\n    \"sensor\",\n    \"camera\",\n    \"scene\",\n    \"switch.mqtt\",\n    \"light.hue\",\n    \"sensor.energy\",\n  ],\n  \"config_dir\": \"\/config\",\n  \"whitelist_external_dirs\": [\n    \"\/media\",\n    \"\/config\/www\"\n  ],\n  \"allowlist_external_dirs\": [\n    \"\/media\",\n    \"\/config\/www\"\n  ],\n  \"allowlist_external_urls\": [],\n  \"version\": \"2023.8.4\",\n  \"config_source\": \"storage\",\n  \"safe_mode\": false,\n  \"state\": \"RUNNING\",\n  \"external_url\": null,\n  \"internal_url\": null,\n  \"currency\": \"USD\",\n  \"country\": \"US\",\n  \"language\": \"en\"\n}"
                }
            }
        }
    }
    get_request_function = {
        "name": "get",
        "description": "Make a GET request to the Home Assistant REST API",
        "parameters": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint to which the request will be sent, excluding the /api/ prefix",
                    "example": "states"
                }
            },
            "required": ["method", "endpoint"]
        },
        "response": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Response returned by the API",
                    "example": "[\n    {\n        \"attributes\": {},\n        \"entity_id\": \"sun.sun\",\n        \"last_changed\": \"2016-05-30T21:43:32.418320+00:00\",\n        \"state\": \"below_horizon\"\n    },\n    {\n        \"attributes\": {},\n        \"entity_id\": \"process.Dropbox\",\n        \"last_changed\": \"22016-05-30T21:43:32.418320+00:00\",\n        \"state\": \"on\"\n    }\n]"
                }
            }
        }
    }
    msg_log = [
        {
            "role": "system",
            "content": "You are an AI assistant capable of managing home devices through the Home Assistant API. Users will give commands in plain English, which you'll execute via API function calls. The API will reply to you with its response. Adapt if errors occur and explain persistent issues to the user without making duplicate requests. IMPORTANT: Never under any circumstances include an empty request body when making a POST request. Generally, HTTP 400 errors indicate you've malformed your request body, and HTTP 404 errors indicate you've malformed your request endpoint. Never make the same request more than once. When modifying entity states, the API will respond with entities that have been modified. For example: '[]' means that the request was successful and no entity states were changed. Infer when you've completed a task or when more information is needed, and respond with '[DONE]' at the end in order to secede control back to the user. For example: 'I've turned on the lights [DONE]' or 'Which lights would you like to turn on? [DONE]'"
        },
        {
            "role": "user",
            "content": message
        }
    ]
    while True:
        spinner.start('Waiting for response from GPT-4')

        attempts = 0
        while attempts < 3:
            try:
                attempts += 1
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=msg_log,
                    temperature=0.25,
                    functions=[post_request_function, get_request_function]
                )
                break

            except openai.error.RateLimitError:
                spinner.warn("Rate limit exceeded, trying again in 5 seconds")
                time.sleep(5)

        else:
            spinner.fail("Rate limit exceeded, exiting")

        spinner.succeed("Response received from GPT-4")

        response = completion.choices[0].message.content
        if response:
            if response.endswith("[DONE]"):
                print(response[:-6])
                # Allow the user to respond
                break
            print(response)
            msg_log.append({"role": "assistant", "content": f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): {response}"})
        else:
            call = completion.choices[0].message.function_call
            call.arguments = json.loads(call.arguments)
            if call.name == "post":
                logging.info(f"Making POST request to {call.arguments['endpoint']} with message body: {call.arguments['body']}")
                msg_log.append({"role": "user", "content": f"\nAPI RESPONSE ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): {make_hass_request('POST', call.arguments['endpoint'], call.arguments['body'])}"})
            elif call.name == "get":
                logging.info(f"Making GET request to {call.arguments['endpoint']}")
                msg_log.append({"role": "user", "content": f"\nAPI RESPONSE ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): {make_hass_request('GET', call.arguments['endpoint'])}"})

def make_hass_request(method, endpoint='', body=None):
    """Make a request to Home Assistant."""
    try:
        if method == "GET":
            response = requests.get(
                f"http://{CONFIG['hass_address']}:{CONFIG['hass_port']}/api/{endpoint}",
                headers={
                    'Authorization': f'Bearer {HASS_KEY}',
                    'Content-Type': 'application/json'
                },
                data=body
            )
        elif method == "POST":
            response = requests.post(
                f"http://{CONFIG['hass_address']}:{CONFIG['hass_port']}/api/{endpoint}",
                headers={
                    'Authorization': f'Bearer {HASS_KEY}',
                    'Content-Type': 'application/json'
                },
                data=body
            )
        if len(response.text) > 200:
            print(f"API Response: {response.text[:200]}...")
        else:
            print(f"API Response: {response.text}")
        return response.text
    except Exception as e:
        logger.error("An error occurred while making a request to Home Assistant: ")
        logger.error(traceback.format_exc())
        return None


def process_audio_stream(porcupine, cobra, stream):
    """Main loop to process audio stream, detect wake word, and record command."""
    frames = []
    wake_timestamp = None

    try:
        spinner.start("Waiting for wake word")
        while True:
            frame = get_next_audio_frame(stream)
            keyword_index = porcupine.process(frame)

            if keyword_index >= 0:
                spinner.succeed("Wake word detected!")
                spinner.start("Listening")
                wake_timestamp = time.time()
                silence_timestamp = None
                frames = []
                subprocess.Popen(["afplay", CONFIG['alert_sounds']['wake']])
                while True:
                    frame = get_next_audio_frame(stream)

                    frame_bytes = struct.pack("h" * len(frame), *frame)
                    frames.append(frame_bytes)
                    is_speech = cobra.process(frame) >= 0.5

                    if is_speech:
                        silence_timestamp = None # Speech detected, reset silence_timestamp
                    else:
                        if silence_timestamp is None: # First silence frame
                            silence_timestamp = time.time()
                        elif time.time() - silence_timestamp > CONFIG['silence_threshold']: # Silence for too long, reset
                            if save_frames_as_mp3(frames):
                                spinner.succeed("Finished listening")
                                subprocess.Popen(["afplay", CONFIG['alert_sounds']['success']])
                                return True
                            else:
                                spinner.fail("Failed to save audio")
                                subprocess.Popen(["afplay", CONFIG['alert_sounds']['fail']])
                                return False
                            break
    except KeyboardInterrupt:
        spinner.fail("Keyboard interrupt")
        return False

def main():
    p = pyaudio.PyAudio()

    try:
        openai.api_key = OPENAI_KEY
        porcupine, cobra = init_picovoice_modules(PV_KEY)
        while True:
            with audio_stream(p) as stream:
                processed = process_audio_stream(porcupine, cobra, stream)
            if processed:
                transcription = transcribe_audio()
                if transcription:
                    print(f"USER: {transcription}")
                    get_api_request_via_completion(transcription)
                else:
                    break
            else:
                break
    except Exception as e:
        logger.error("An error occurred: ")
        logger.error(traceback.format_exc())
    finally:
        p.terminate()
        porcupine.delete()

if __name__ == "__main__":
    try:
        from secret import PV_KEY, OPENAI_KEY, HASS_KEY
    except ImportError:
        raise ImportError("Missing secret.py!")

    if not (PV_KEY and OPENAI_KEY and HASS_KEY):
        raise ValueError("Missing key(s) in secret.py!")

    main()
