import itertools
import keyring
import sounddevice as sd
import soundfile as sf
from io import BytesIO
from gtts import gTTS
import pyaudio
import wave
import numpy as np
from faster_whisper import WhisperModel
import openai
import os
import nltk
import logging
import re
import nltk.data
from lingua import Language, LanguageDetectorBuilder
from elevenlabslib import *
from elevenlabslib.helpers import *
import atexit
import signal
import string
import json
import sys
import argparse
from num2words import num2words
import time


SETTINGS_FILE = "settings.dat"

def contains_sentence_ending(chunk_text):
    sentence_ending_pattern = r'[.?!]'
    numbered_list_pattern = r'\d+\.'
    ip_address_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    abbreviation_pattern = r'\b[A-Za-z]\.'

    if re.search(sentence_ending_pattern, chunk_text) and \
            not re.search(numbered_list_pattern, chunk_text) and \
            not re.search(ip_address_pattern, chunk_text) and \
            not re.search(abbreviation_pattern, chunk_text):
        return True
    else:
        return False

def sleep_ms(milliseconds):
    start_time = time.perf_counter()
    desired_time = milliseconds / 1000  # Convert milliseconds to seconds
    elapsed_time = 0
    
    while elapsed_time < desired_time:
        elapsed_time = time.perf_counter() - start_time

def clean_up(*args):
    global aud_audio
    aud_audio.terminate()
    print('PyAudio terminated')
    sys.exit()

atexit.register(clean_up)
signal.signal(signal.SIGINT, clean_up)

languages = [Language.ARABIC, Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.HINDI, Language.ITALIAN, Language.LATIN, Language.MAORI, Language.POLISH, Language.PORTUGUESE,Language.SPANISH, Language.CHINESE, Language.JAPANESE, Language.KOREAN, Language.RUSSIAN, Language.UKRAINIAN, Language.ARABIC]
logging.basicConfig(filename='logfile.log', level=logging.INFO)
language_isocodes = {
    Language.ARABIC: 'ar',
    Language.ENGLISH: 'en',
    Language.FRENCH: 'fr',
    Language.GERMAN: 'de',
    Language.HINDI: 'hi',
    Language.ITALIAN: 'it',
    Language.LATIN: 'la',
    Language.MAORI: 'mi',
    Language.POLISH: 'pl',
    Language.PORTUGUESE: 'pt',
    Language.SPANISH: 'es',
    Language.CHINESE: 'zh',
    Language.JAPANESE: 'ja',
    Language.RUSSIAN: 'ru',
    Language.UKRAINIAN: 'ua',
    Language.ARABIC: 'ar',
    Language.KOREAN: 'ko'
}

def load_presets(file_name):
    try:
        with open(file_name, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Presets file not found. Loading default preset.")
        return [{
            'name': 'assistant',
            'prompt': 'you are a helpful assistant.'
        }]
    # split presets by two consecutive newline characters
    presets = content.split('\n\n')

    presets_list = []
    for preset in presets:
        # each preset is a dictionary with name and prompt as keys
        preset_dict = {}

        # split by newline to get name and prompt separately
        lines = preset.split('\n')

        # extract name, removing 'name: '
        name_line = lines[0].strip()
        name = re.sub(r'^name:\s*', '', name_line, flags=re.IGNORECASE)
        preset_dict['name'] = name

        # extract prompt, removing 'prompt:' from the second line, and include the rest of the lines
        prompt_lines = [line.strip() for line in lines[1:]]
        prompt = ' '.join(prompt_lines)
        prompt = re.sub(r'^prompt:\s*', '', prompt, count=1, flags=re.IGNORECASE)
        preset_dict['prompt'] = prompt

        presets_list.append(preset_dict)

    return presets_list

def get_preset(user_string, presets):
    user_string = user_string.lower()  # Convert the user string to lowercase for case insensitivity

    for preset in presets:
        preset_name = preset['name'].lower()
        if user_string == preset_name:
            return preset

    return presets[0] if presets else None

def create_preset(presets_file):
    # Load existing presets
    presets = load_presets(presets_file)

    # Prompt the user for name and prompt
    if settings["handsfreeornot"]["state"] == False:
        name = input("Enter the preset name: ")
        prompt = input("Enter the preset prompt: ")
    else:
        print("Please speak the name of the preset.")
        name=join(ch for ch in recordandtranscribe() if ch.isalnum() or ch.isspace())
        print("Please speak the prompt of the preset.")
        prompt=recordandtranscribe()
    # Create a new preset dictionary
    new_preset = {
        'name': name,
        'prompt': prompt
    }

    # Add the new preset to the presets list
    presets.append(new_preset)

    # Save the updated presets to a new file
    new_presets_file = presets_file.replace('.txt', '_new.txt')
    with open(new_presets_file, 'w') as f:
        for preset in presets:
            f.write(f"name: {preset['name']}\n")
            f.write(f"prompt: {preset['prompt']}\n\n")

    print("New preset created and saved successfully!")

def yes_no_question(question):
    while True:
        user_response = input(question + " (y/n): ").lower()
        if user_response in ["y", "yes"]:
            return True
        elif user_response in ["n", "no"]:
            return False
        else:
            print("Invalid input. Please enter y/n or yes/no.")

def display_elevenlabs_voices(voices):
    print("\nAvailable Voices:")
    for i, voice in enumerate(voices, start=1):
        print(f"Press {i} for {voice.initialName}")

def get_elevenlabs_user_voice_choice(voices):
    global settings
    global voice_names
    choice = None

    while choice is None or (not isinstance(choice, int) or choice < 1 or choice > len(voices)) and (not isinstance(choice, str) or choice not in voice_names):
        try:
            if settings["handsfreeornot"]["state"] == True:
                user_input = ''.join(ch for ch in recordandtranscribe() if ch.isalnum() or ch.isspace()).lower()
                for voice_name in voice_names:
                    if voice_name in user_input:
                        choice = voice_names[voice_name]
                        break
                else:
                    try:
                        # try to convert the transcribed string to integer directly
                        choice = int(user_input)
                    except ValueError:
                        # if direct conversion fails, try to convert from word to number
                        try:
                            user_input = word_to_num(user_input)
                            choice = int(user_input)
                        except ValueError:
                            print("Invalid input. Please enter a number.")
            else:
                user_input = input("Enter your choice: ")
                choice = int(user_input)

            if (isinstance(choice, int) and (choice < 1 or choice > len(voices))) or (isinstance(choice, str) and choice not in voice_names):
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or voice name.")

    return choice

def split_text():
    """
    Split this file into multiple files. Each file must be under chunk_range, o
    and the break must occur between sentences.
    :return:
    """
    global chat_response
    global CHUNK_RANGE
    text = chat_response
    # Strip all tab characters
    text = text.replace('\t', '')

    # Condense multiple space characters down to 1
    text = re.sub(r' +', ' ', text)

    # Condense multiple new line characters down to 1
    text = re.sub(r'\n+', '\n', text)
    # Used NLTK to split the text into sentences
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(text)

    chunks = []
    chunk = ""
    for i, se in enumerate(sentences):
        if len(chunk) + len(se) > CHUNK_RANGE:
            chunks.append(chunk)
            chunk = se

        elif len(chunk) + len(se) < CHUNK_RANGE:
            chunk = chunk + " " + se

        elif len(chunk) + len(se) == CHUNK_RANGE:
            chunk = chunk + "" + se
            chunks.append(chunk)
            chunk = ""
        else:
            raise Exception("The limit condition exceed")

        # To handle last paragraph,
        if i == len(sentences) - 1:
            chunks.append(chunk)
    return chunks

def get_elevenlabs_voice():
    global voice
    global voices
    global user
    global voice_names
    if voices == "":
        voices = user.get_available_voices()  # get the voices that can be used.
        voice_names = {''.join(ch for ch in voice.initialName if ch.isalnum() or ch.isspace()).lower(): i+1 for i, voice in enumerate(voices)}
    display_elevenlabs_voices(voices)
    user_voice_choice = get_elevenlabs_user_voice_choice(voices)
    voice = voices[user_voice_choice - 1]

def recordandtranscribe():
    global model
    global aud_audio
    global stream
    aud_FORMAT = pyaudio.paInt16
    aud_CHANNELS = 1
    aud_RATE = 44100
    aud_CHUNK = 1024
    aud_FILENAME = "tempoutput.wav"
    aud_SILENCE_THRESHOLD = 1000  # adjust this as needed
    SILENCE_CHUNKS = 60  # adjust this as needed
    # start Recording
    stream = aud_audio.open(format=aud_FORMAT, channels=aud_CHANNELS,
    rate=aud_RATE, input=True,
    frames_per_buffer=aud_CHUNK)
    aud_frames = []
    silence_chunk_counter = 0
    print(f"Waiting for speech... \n" )
    while True:
        data = stream.read(aud_CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        if np.abs(audio_data).mean() >= aud_SILENCE_THRESHOLD:  # Speech detected
            aud_frames.append(data)
            break
    print(f"Recording...\n")
    while True:
        data = stream.read(aud_CHUNK)
        aud_frames.append(data)

        # check for silence here by converting the data to numpy arrays
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.abs(audio_data).mean() < aud_SILENCE_THRESHOLD:  # this threshold depends on your microphone
            silence_chunk_counter += 1
        else:
            silence_chunk_counter = 0  # Reset counter if non-silent chunk detected
        if silence_chunk_counter >= SILENCE_CHUNKS:  # Enough silence detected
            print(f"Silence detected, stopping recording \n")
            break
    # stop Recording
    stream.stop_stream()
    stream.close()

    waveFile = wave.open(aud_FILENAME, 'wb')
    waveFile.setnchannels(aud_CHANNELS)
    waveFile.setsampwidth(aud_audio.get_sample_size(aud_FORMAT))
    waveFile.setframerate(aud_RATE)
    waveFile.writeframes(b''.join(aud_frames))
    waveFile.close()

    print(f"Recording saved to {aud_FILENAME}.  Transcribing text... \n")
    if settings["whisperapiorlocal"]["state"] == True:
        try:
            audio_file = open(aud_FILENAME, "rb")
            message = openai.Audio.transcribe("whisper-1", audio_file).text
        except Exception as e:
            settings["whisperapiorlocal"]["state"] = False
            print(f"An error occurred, switching to local mode.: {str(e)}")

    if settings["whisperapiorlocal"]["state"] == False:
        segments, info = model.transcribe(aud_FILENAME, beam_size=5, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500))
        print("whisper Detected language '%s' with probability %f" % (info.language, info.language_probability))
        logging.info("whisper Detected language {info.language} with probability {info.language_probability}" )
        segments, _ = model.transcribe(aud_FILENAME)
        segments = list(segments)  # The transcription will actually run here.
        texts = []
        for textsegment in segments:
            texts.append(textsegment.text)
        message = ' '.join(texts)
    #os.remove(aud_FILENAME)

    print(f"user: {message}")
    return message
# Settings with their corresponding questions
settings = {
    "gpt4ornot": {
        "question": "Would you like to use GPT4? Otherwise, it will use GPT-3.5",
        "state": None
    },
    "handsfreeornot": {
        "question": "Do you want to use hands-free mode? Hands-free mode will force you to control the program exclusively via voice.",
        "state": None
    },
    "useelevenlabs": {
        "question": "Do you want to use Elevenlabs or not? If you press 'no', the program will use Google TTS. Be aware that Google TTS will be used for certain languages as a fallback.",
        "state": None
    },
    "whisperapiorlocal": {
        "question": "Do you want to use Whisper API? If you say no, whisper will be run locally. Using the API is faster but comes with a cost.",
        "state": None
    },
    "gpuornot": {
        "question": "In case Whisper needs to be run locally, would you like to try to use the GPU?",
        "state": None
    }
}

def load_settings():
    """Loads settings from the settings file"""
    if os.path.isfile(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as file:
            loaded_settings = json.load(file)
    else:
        loaded_settings = ask_questions(settings)
    save_settings(loaded_settings)
    return loaded_settings

def save_settings(settings):
    """Saves settings to the settings file"""
    with open(SETTINGS_FILE, "w") as file:
        json.dump(settings, file)

def ask_questions(settings):
    """Asks questions to the user and saves the responses as settings"""
    for setting, value in settings.items():
        if value["state"] is None:
            settings[setting]["state"] = yes_no_question(value["question"])
    return settings

def get_key(keytype):
    key = ""
    key = keyring.get_password("speakingGPT", keytype)
    if key is not None:
        # Use the API key
        print(f"An API key has been found and will be used. \n")
    else:
        # API key not found
        key = input(f"What is your {keytype} api key?")
        keyring.set_password("speakingGPT", keytype, key)
    return key

def load_preset():
    global presets
    global settings
    global messagerole
    print(f"list of presets. \n")
    for preset in presets:
        print(f'{preset["name"]} \n')
    is_first_time = True
    while True:
        if is_first_time == False:
            print("preset not found")
        if settings["handsfreeornot"]["state"]:
            print(f"please speak the name of the preset. \n")
            name = ''.join(ch for ch in recordandtranscribe() if ch.isalnum()).lower()
        else:
            name = ''.join(ch for ch in input("Enter the name of the preset you want to load: ") if ch.isalnum()).lower()
        for preset in presets:
            preset_name = preset['name']
            # remove spaces and convert to lowercase
            preset_name = ''.join(ch for ch in preset_name if ch.isalnum()).lower()
            if name == preset_name:
                messagerole = process_preset(preset["prompt"])
                return
        is_first_time = False

def setup():
    global CHUNK_RANGE
    global voice
    global voices
    global user
    global model
    global aud_audio
    global settings
    global aud_audio
    global detector
    global presets
    global messagerole
    messagerole = ""
    voices = ""
    CHUNK_RANGE = 600
    presets = load_presets("presets.dat")
    parser = argparse.ArgumentParser(description="This app allows you to converse with chat GPT by voice, using faster_whisper as input and a choice between ElevenLabs and google TTS for output.")
    parser.add_argument("--preset", type=str, default="assistant", help="preset name here, default is assistant")
    args = parser.parse_args()
    openai.api_key = get_key("openai")
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    model_size = "large-v2"
    # Check if settings file exists
    settings = load_settings()
    # Continue with the rest of the code
    model = WhisperModel(model_size, device="auto" if settings["gpuornot"]["state"] else "cpu")
    # initialize audio
    aud_audio = pyaudio.PyAudio()
    if settings["useelevenlabs"]["state"] == True:
        for _ in range(5):
            try:
                elevenkey = get_key("elevenlabs")
                user = ElevenLabsUser(elevenkey)
                break
            except ValueError as e:
                if "Invalid API Key!" in str(e):
                    print("The ElevenLabs API Key you provided is invalid.")
                    keyring.delete_password("speakingGPT", "elevenlabs")
                    elevenkey = get_key("elevenlabs")
                    continue
                else:
                    raise
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    print("Unauthorized. Please check your ElevenLabs API Key.")
                    keyring.delete_password("speakingGPT", "elevenlabs")
                    elevenkey = get_key("elevenlabs")
                    continue
                else:
                    raise
        else:
            print("Failed to initialize ElevenLabsUser after several attempts.")

        get_elevenlabs_voice()
    nltk.download("punkt")
    preset_to_use = get_preset(args.preset, presets)
    messagerole = process_preset(preset_to_use["prompt"])
    print(f'using { preset_to_use["name"] } preset. \n')

def process_preset(template):
    # Load the NLTK sentence tokenizer
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # Split the template into sentences
    sentences = sent_detector.tokenize(template)

    # Keep track of the placeholders and their corresponding values
    placeholder_values = {}

    # Process each sentence
    for i in range(len(sentences)):
        # Find placeholders in this sentence
        placeholders = re.findall(r'{{(.*?)}}', sentences[i])

        # Replace each placeholder with user input
        for placeholder in placeholders:
            if placeholder not in placeholder_values:
                # If it's a new placeholder, show the user the sentence context and prompt for a value
                print(sentences[i])

                # Prompt the user for a replacement
                if settings["handsfreeornot"]["state"] == False:
                    value = input(f"Please enter a value for '{placeholder}': ")
                else:
                    print(f"Please enter a value for '{placeholder}': ")
                    value = ''.join(ch for ch in recordandtranscribe() if ch.isalnum() or ch.isspace())
                placeholder_values[placeholder] = value

            # Replace the placeholder with the user's input
            sentences[i] = sentences[i].replace(f"{{{{{placeholder}}}}}", placeholder_values[placeholder])

        # Print the sentence with placeholders filled in
        print(sentences[i])

    # Join the sentences back together
    return ' '.join(sentences)

def play_audio_bytes(audio_data):
    # Create a BytesIO object and write the audio data to it
    audio_stream = audio_data
    audio_stream.seek(0)  # Reset the stream position to the beginning

    # Get audio file information using soundfile
    info = sf.info(audio_stream)

    # Read audio data using soundfile
    audio_data, _ = sf.read(audio_stream, dtype='float32')

    # Play the audio data
    sd.play(audio_data, info.samplerate)
    sd.wait()

def convert_numbers_to_words(text, lang):
    return re.sub(r'\b\d+\b(?=\W|\b)', lambda m: num2words(int(m.group()), lang=lang), text)

        
def extract_text_from_chunks(chunk_iterator):
    for chunk in chunk_iterator:
        if 'choices' in chunk and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
            yield chunk['choices'][0]['delta']['content']
def store_and_yield(iterator, storage_list):
    for item in iterator:
        storage_list.append(item)
        yield item

def voice_response(text_iterator):
    global detector
    global voice
    global messages

    # List to store chunks as they're processed
    chunks = []

    canuseelevenlabs = settings["useelevenlabs"]["state"] == True

    if canuseelevenlabs:
        try:
            modelid = 'eleven_multilingual_v1'
            playbackOptions = PlaybackOptions(runInBackground=False) 
            generationOptions = GenerationOptions(model_id=modelid, latencyOptimizationLevel = 3) 

            # Wrap the iterator to store its values
            storing_iterator = store_and_yield(text_iterator, chunks)

            result = voice.generate_stream_audio_v2(storing_iterator, playbackOptions, generationOptions)
        except Exception as e:
            print(f"An error occurred: {e}")
            canuseelevenlabs = False
    # If ElevenLabs wasn't used, process the accumulated text with Google TTS
    if not canuseelevenlabs:
        accumulated_text = []
        for chat_response in text_iterator:
            chunks.append(chat_response)
            accumulated_text.append(chat_response)

        full_text = ''.join(accumulated_text)
        whatlang = detector.detect_language_of(full_text)
        tts = gTTS(full_text, lang=language_isocodes.get(whatlang, 'en'))
        mp3fp = BytesIO()
        tts.write_to_fp(mp3fp)
        play_audio_bytes(mp3fp)

    return ''.join(chunks)  # Return the full text

def word_to_num(word):
    word = ''.join(ch for ch in word if ch.isalnum())  # Remove non-alphanumeric characters
    word = word.lower()
    # Dictionaries for word-number pairs in six languages
    word_num_dict = {
        # English
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
        "twentyone": 21, "twentytwo": 22, "twentythree": 23, "twentyfour": 24,
        "twentyfive": 25, "twentysix": 26, "twentyseven": 27, "twentyeight": 28,
        "twentynine": 29, "thirty": 30,

        # Spanish
        "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
        "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
        "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
        "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
        "veintiuno": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
        "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
        "veintinueve": 29, "treinta": 30,

        # German
        "null": 0, "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
        "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
        "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15,
        "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19, "zwanzig": 20,
        "einundzwanzig": 21, "zweiundzwanzig": 22, "dreiundzwanzig": 23, 
        "vierundzwanzig": 24, "fünfundzwanzig": 25, "sechsundzwanzig": 26, 
        "siebenundzwanzig": 27, "achtundzwanzig": 28, "neunundzwanzig": 29, 
        "dreißig": 30,

        # French
        "zéro": 0, "un": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
        "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
        "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15,
        "seize": 16, "dixsept": 17, "dixhuit": 18, "dixneuf": 19, "vingt": 20,
        "vingtetun": 21, "vingtdeux": 22, "vingttrois": 23, "vingtquatre": 24,
        "vingtcinq": 25, "vingtsix": 26, "vingtsept": 27, "vingthuit": 28,
        "vingtneuf": 29, "trente": 30,

        # Italian
        "zero": 0, "uno": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5,
        "sei": 6, "sette": 7, "otto": 8, "nove": 9, "dieci": 10,
        "undici": 11, "dodici": 12, "tredici": 13, "quattordici": 14, "quindici": 15,
        "sedici": 16, "diciassette": 17, "diciotto": 18, "diciannove": 19, "venti": 20,
        "ventuno": 21, "ventidue": 22, "ventitre": 23, "ventiquattro": 24,
        "venticinque": 25, "ventisei": 26, "ventisette": 27, "ventotto": 28,
        "ventinove": 29, "trenta": 30,

        # Latin
        "nullus": 0, "unus": 1, "duo": 2, "tres": 3, "quattuor": 4, "quinque": 5,
        "sex": 6, "septem": 7, "octo": 8, "novem": 9, "decem": 10,
        "undecim": 11, "duodecim": 12, "tredecim": 13, "quattuordecim": 14, 
        "quindecim": 15, "sedecim": 16, "septendecim": 17, "duodeviginti": 18, 
        "undeviginti": 19, "viginti": 20, "vigintiunus": 21, "vigintiduo": 22, 
        "vigintitres": 23, "vigintiquattuor": 24, "vigintiquinque": 25, 
        "vigintisex": 26, "vigintiseptem": 27, "vigintiocto": 28,
        "vigintinovem": 29, "triginta": 30,
    }

    try:
        return str(word_num_dict[word.lower()])
    except KeyError:
        return "-1"

def save_conversation(file_name, messages):
    try:
        with open(file_name, 'w') as outfile:
            json.dump(messages, outfile)
    except IOError as e:
        print(f"An error occurred while trying to write to the file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def load_conversation(file_name):
    if os.path.exists(file_name):
        try:
            with open(file_name, 'r') as infile:
                messages = json.load(infile)
            return messages
        except IOError as e:
            print(f"An error occurred while trying to read the file: {e}")
        except ValueError as e:
            print(f"Could not decode JSON: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print("The file does not exist. Please check your file name and path.")

def print_conversation(messages):
    try:
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")
    except KeyError as e:
        print(f"Message is missing some keys: {e}")
    except TypeError as e:
        print(f"An error occurred while trying to print the conversation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def chat():
    global chat_response
    global settings
    global messages
    global messagerole
    print(f"Welcome to Speaking GPT!  Remember, all generations are done using an AI system, and the conversation produced is not human output. \n")
    setup()
    messages = [
    {"role": "system", "content": messagerole, "historyids": []},
]

    while True:
        message = ""
        if settings["handsfreeornot"]["state"] == False:
            message = input("User: ")
        if message.strip() == "rec" or settings["handsfreeornot"]["state"] == True:
            message = recordandtranscribe()
        if message.strip() == "chv" or "change voice" in message.lower():
            if settings["useelevenlabs"]["state"] == True:
                get_elevenlabs_voice()
            else:
                print(f"unfortunately, changing voices in google TTS is not supported. \n")
            continue
        lowermessage = message.lower()
        clean_message = ''.join(ch for ch in lowermessage if ch.isalnum())
        if "cp" == clean_message or "createpreset" in clean_message:
            create_preset("presets.dat")
            continue
        if "lc" == clean_message or "loadconversation" in clean_message:
            filenametoload = input("file name?")
            messages = load_conversation(filenametoload)
            print_conversation(messages)
            continue
        if "sc" == clean_message or "saveconversation" in clean_message:
            filenametosave = input("file name?")
            save_conversation(filenametosave, messages)
            print (f"conversation successfully saved to {filenametosave} \n" )
            continue
        if "chf" == clean_message or "changehandsfree" in clean_message:
            settings["handsfreeornot"]["state"] = not settings["handsfreeornot"]["state"]
            save_settings(settings)
            if settings["handsfreeornot"]["state"]:
                print("hands free on\n")
            else:
                print("hands free off\n")
            continue
        if "lp" == clean_message or "loadpreset" in clean_message:
            load_preset()
            messages = [
    {"role": "system", "content": messagerole, "historyids": []},
]
            print(f"preset loaded. clearing conversation. \n")
            continue
        messages.append({"role": "user", "content": message, "historyids": []})
        print(f"processing \n")
        sanitized_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

        for _ in range(10):  # replace 3 with desired number of attempts
            try:
                response = openai.ChatCompletion.create(
            model="gpt-4" if settings["gpt4ornot"]["state"] else "gpt-3.5-turbo-16k",
            messages=sanitized_messages,
            stream=True
        )
                break
            except openai.error.InvalidRequestError as e:
                if "does not exist" in str(e):
                    print("Switching to GPT-3.5 Turbo due to invalid access to GPT-4.")
                    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=sanitized_messages,
                stream=True
            )
                    settings["gpt4ornot"]["state"] = False
                    save_settings(settings)
                    break
                else:
                    raise
            except openai.error.AuthenticationError as e:
                print("There was an error with the API key. Please check the key and try again.")
                keyring.delete_password("speakingGPT", "openai")
                openai.api_key = get_key("openai")
                # prompt for a new API key here, then continue the loop
                # you can replace 'continue' with 'break' if you want to stop after first failure
                continue
            except openai.error.RateLimitError as e:
                print("The model is busy. retrying in 15 seconds.")
                time.sleep(15)
                continue
        else:
            print("Failed to create a chat completion after several attempts.")
            file_to_save = input("please enter a file name to save your conversation and try again later.")
            save_conversation(file_to_save, messages)
            sys.exit()
        text_iterator = extract_text_from_chunks(response)
        full_response = voice_response(text_iterator)        print(f"ai: {full_response}")        messages.append({"role": "assistant", "content": full_response, "historyids": []})
if __name__ == "__main__":
    print("Starting the AI...")
    chat()



