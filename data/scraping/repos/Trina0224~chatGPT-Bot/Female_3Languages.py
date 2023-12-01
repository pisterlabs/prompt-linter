import os
from google.cloud import texttospeech_v1
import io
import time
import threading
from google.oauth2 import service_account
#from google.cloud import speech
from google.cloud import speech_v1p1beta1 as speech
from aiy.board import Board,Led
from aiy.voice.audio import AudioFormat, play_wav, record_file, Recorder
import os
import openai
import pygame

User_Language_Code=''

def check_button_press():
    with Board() as board:
        #board.led.state = Led.ON
        while True:
            board.button.wait_for_press()
            pygame.mixer.music.stop()
            #board.led.state = Led.OFF


def text_to_speech(input_text, output_filename="output.mp3"):

    # Assuming the credentials file is in the same directory
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "Your Google text2speech.json"

    # Instantiates a client
    client = texttospeech_v1.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech_v1.SynthesisInput(text=input_text)

    # Determine the voice parameters based on User_Language_Code
    if 'en' in User_Language_Code:
        language_code = "en-US"
        voice_name = "en-US-Wavenet-F"
    elif 'jp' in User_Language_Code:
        language_code = "ja-JP"
        voice_name = "ja-JP-Wavenet-B"
    else:  # default to 'cmn' settings
        language_code = "cmn-TW"
        voice_name = "cmn-TW-Wavenet-A"

    # Voice parameters
    voice = texttospeech_v1.VoiceSelectionParams(
        language_code=language_code, 
        name=voice_name
    )

    # Audio format
    audio_config = texttospeech_v1.AudioConfig(
        audio_encoding=texttospeech_v1.AudioEncoding.MP3,
        speaking_rate=1.3
    )

    # Make the API call
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Save the response to an output file
    with open(output_filename, 'wb') as out:
        out.write(response.audio_content)

    return output_filename

    #return output_filename

def play_mp3(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    # Start the button monitoring thread right after
    button_thread = threading.Thread(target=check_button_press)
    button_thread.start()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)









def recognize_audio(filename):
    global User_Language_Code
    with io.open(filename, 'rb') as f:
        content = f.read()
        audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        audio_channel_count=2,
        sample_rate_hertz=44100,
        language_code = 'zh-TW',
        alternative_language_codes = ['en-US','ja-JP']
    )
    response = client.recognize(config=config, audio=audio)
    os.remove(filename)
    if response.results:
        User_Language_Code = response.results[0].language_code
        print('The user is speaking '+ User_Language_Code + '\n' )
        #print(response)
        print('{0}: {1}\n'.format(response.results[0].language_code, response.results[0].alternatives[0].transcript))
        return response.results[0].alternatives[0].transcript
    return ""

def record_and_recognize():
    filename = 'recording.wav'
    with Board() as board:
        print('Press button to start recording.')
        board.led.state = Led.ON
        board.button.wait_for_press()

        done = threading.Event()
        board.button.when_pressed = done.set

        def wait():
            start = time.monotonic()
            while not done.is_set():
                duration = time.monotonic() - start
                print('Recording: %.02f seconds [Press button to stop]' % duration)
                time.sleep(0.5)

        record_file(AudioFormat.CD, filename=filename, wait=wait, filetype='wav')
        board.led.state = Led.OFF
        print('Sending audio for recognition...')
        recognized_text = recognize_audio(filename)
        return recognized_text




# Google Cloud Speech-to-Text client setup
client_file = 'Your Google speech2text.json'
credentials = service_account.Credentials.from_service_account_file(client_file)
client = speech.SpeechClient(credentials=credentials)

API_KEY = 'Your OpenAI Key'
openai.api_key = API_KEY

#messages = [ {"role": "system", "content": 
 #             "You are a intelligent assistant."} ]

#model_id = 'gpt-4'
model_id = 'gpt-3.5-turbo'

def chatgpt_conversation(conversation_log):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation_log
    )

    conversation_log.append({
        'role': response.choices[0].message.role, 
        'content': response.choices[0].message.content.strip()
    })
    return conversation_log

conversations = []
# role: system, user, assistant
conversations.append({'role': 'system', 'content': 'You are a intelligent assistant.'})
conversations = chatgpt_conversation(conversations)

print('{0}: {1}\n'.format(conversations[-1]['role'].strip(), conversations[-1]['content'].strip()))
#filename = text_to_speech('{0}: {1}'.format(conversations[-1]['role'].strip(), conversations[-1]['content'].strip()))
#print(f"Audio saved to: {filename}")
play_mp3("greeting_jp.mp3")


while True:
    prompt = record_and_recognize()
    conversations.append({'role': 'user', 'content': prompt})
    conversations = chatgpt_conversation(conversations)
    print()
    print('{0}: {1}\n'.format(conversations[-1]['role'].strip(), conversations[-1]['content'].strip()))
    #print(conversations[-1]['content'].strip())

    filename = text_to_speech(conversations[-1]['content'].strip())
    #print(f"Audio saved to: {filename}")
    #with Board() as board:
    #    board.led.state = Led.ON
    play_mp3(filename)
