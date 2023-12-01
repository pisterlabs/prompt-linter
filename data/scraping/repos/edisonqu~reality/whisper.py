#! python3.7

import argparse
import io
import os
from chromadb.utils.embedding_functions import CohereEmbeddingFunction
import dotenv
import speech_recognition as sr
import openai

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
import chromadb
from chromadb.config import Settings
from embeddings import insert_to_db
from database import database_initialization_and_collection

phrase_time = None

MAX_BYTE_SIZE = 1e7
TEMP_DIR = ".note_files/"


def record(cohere_ef,collection):
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--use_openai_api", action='store_true',
                        help="Using OPENAI_API_KEY enviromental variable")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=6,
                        help="How real time the recording is in seconds .", type=float)
    parser.add_argument("--phrase_timeout", default=7,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retreived from the queue.
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(
                        sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv('OPENAI_API_KEY')
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    args.use_openai_api = True

    temp_file = NamedTemporaryFile(suffix='.mp3').name
    # temp_file = "temp_file.wav"
    transcription = []

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)
        global phrase_time
        phrase_time = datetime.utcnow()
        byte_size = sum(len(item) for item in list(data_queue.queue))

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(
        source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    phrase_complete = False

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.

            if phrase_complete:
                print('PHRASE COMPLETE. TRANSCRIBING...')
                with open(temp_file, 'rb') as f:
                    result = openai.Audio.transcribe("whisper-1", f)
                    print(result)

                text = result['text'].strip()
                insert_to_db(collection, text, str(now), cohere_ef)
                transcription.append(text)
                print(text)
                phrase_complete = False
                last_sample = bytes()

            if not data_queue.empty():
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.

                byte_size = sum(len(item) for item in list(data_queue.queue))
                # print(byte_size)
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True

                    while not data_queue.empty():
                        data = data_queue.get()
                        last_sample += data

                    audio_data = sr.AudioData(
                        last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())
                    with open(temp_file, 'w+b') as f:
                        f.write(wav_data.read())
                    # with open(TEMP_DIR + str(now) + ".wav", "wb") as f:
                    #     f.write(wav_data.read())
                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)
    print(transcription)


# if __name__ == "__main__":
#     record()
