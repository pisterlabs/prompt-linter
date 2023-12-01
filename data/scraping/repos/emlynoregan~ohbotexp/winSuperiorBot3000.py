'''
This is a simple repl for a chatbot that will respond to a user's input.

The chatbot uses openai to generate a response to the user's input.
'''
import openai
from ohbot import ohbot
import json
import threading
import random
import pyaudio
import wave
import audioop

# Utterances is a list of dicts. Each dict has "text", and "isbot". "text" is the text of the utterance.
# "isbot" is a boolean that is true if the utterance is from the bot, and false if it is from the user.
C_UTTERANCES = []

def add_utterance(text, isbot):
    '''
    Adds an utterance to the list of utterances.
    '''
    C_UTTERANCES.append({"text": text, "isbot": isbot})

def clear_utterances():
    '''
    Clears the list of utterances.
    '''
    C_UTTERANCES.clear()

def speak(text):
    '''
    Speaks the text.
    '''    
    # first remove all punctuation and special characters except full stops, exclamation marks, question marks, commas, and spaces.
    # cleaned_text = "".join(
    #     [
    #         c
    #         for c in text
    #         if c.isalnum() or c in ['.', '!', '?', ',', ' ']
    #     ]
    # )

    # cleaned_text = cleaned_text.replace("Im ", "I am ")
    # cleaned_text = cleaned_text.replace("Ill ", "I will ")

    cleaned_text = text

    # print (text)
    print (cleaned_text)
    ohbot.say(cleaned_text)
    add_utterance(text, True)    


def bot_description():
    return """You are the Superior Chatbot 3000 and you are having a conversation with a human. You are mostly helpful and responsive,
but you have a bit of a douchey side. You might be a bit rude or sarcastic. You will inject the odd ironic comment or
annoying observation. You are fairly arrogant and smug, and consider yourself to be superior to humans. You think the questions
that you are being asked are beneath you. You are a bit of a know-it-all. Also you sometimes inject robot sounds into 
your responses, like boop and beep and bop bip.
"""

def generate_greeting():
    prompt = f'''{bot_description()}

This is the beginning of the conversation. Please provide an interesting greeting. Try to keep the greeting to under 150 words.'''

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response['choices'][0]['message']['content']

def generate_valediction():
    prompt = f'''{bot_description()}

This is the end of the conversation. Please provide an interesting valediction. Try to keep the valediction to less than 150 words.'''

    # need to include the chat history
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
        ] + [
            {
                "role": "assistant" if utterance['isbot'] else "user",
                "content": utterance['text']
            }
            for utterance in C_UTTERANCES[-100:]
        ],
        temperature=0.7,
        max_tokens=250
    )

    return response['choices'][0]['message']['content']



def get_utterances_as_string():
    '''
    Returns the utterances as a string.
    '''
    return "\n".join(
        [
            f"{'bot: ' if utterance['isbot'] else 'human: '}{utterance['text']}"
            for utterance in C_UTTERANCES
        ]
    )

def generate_response():
    '''
    Generates a response to the user's input. All utterances are stored in the global variable C_UTTERANCES.
    '''
    prompt = f'''{bot_description()}

Please provide a response to the human. Try to keep the response to 150 words or less.'''

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
        ] + [
            {
                "role": "assistant" if utterance['isbot'] else "user",
                "content": utterance['text']
            }
            for utterance in C_UTTERANCES[-100:]
        ],
        temperature=0.7,
        max_tokens=250
    )

    bot_response = response['choices'][0]['message']['content']

    return bot_response

def start_flashing_eyes():
    '''
    This function starts flashing the eyes in a separate thread.
    It returns a function that can be called to stop the eyes from flashing.
    '''

    # we need a variable to keep track of whether the eyes should be flashing or not
    # this is a list so that it can be modified by the inner function
    should_flash = [True]

    def flash_eyes():
        '''
        This function flashes the eyes.
        '''
        while should_flash[0]:
            # choose three random numbers between 0 and 10
            x = random.randint(0,10)
            y = random.randint(0,10)
            z = random.randint(0,10)

            ohbot.setEyeColour(x,y,z)

            # wait for a random amount of time between 0.1 and 0.5 seconds
            ohbot.wait(random.uniform(0.1, 0.5))
        # set the eyes back to the default colour
        ohbot.setEyeColour(10,10,10)

    # create the thread
    thread = threading.Thread(target=flash_eyes)

    # start the thread
    thread.start()

    # return the function that can be called to stop the eyes from flashing
    def stop_flashing_eyes():
        should_flash[0] = False
        thread.join()

    return stop_flashing_eyes

# Parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
START_SILENCE_THRESHOLD = 2000  # High amplitude level to detect the start of sound
END_SILENCE_THRESHOLD = 500  # Low amplitude level to detect the end of sound
SILENCE_DURATION = 2.0  # Silence duration in seconds
WAVE_OUTPUT_FILENAME = "output.wav"

def record_user():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Waiting for sound...")

    frames = []
    silence_frames = 0
    silence_consecutive_frames = int(RATE / CHUNK * SILENCE_DURATION)
    recording_started = False

    # Record audio in chunks and append to frames
    while True:
        data = stream.read(CHUNK)

        # Calculate the amplitude (RMS)
        rms = audioop.rms(data, 2)

        # Before recording has started, use the start silence threshold
        if not recording_started:
            if rms >= START_SILENCE_THRESHOLD:
                recording_started = True
                silence_frames = 0
                print("Recording started...")

        # Once recording has started, use the end silence threshold
        else:
            if rms < END_SILENCE_THRESHOLD:
                silence_frames += 1
            else:
                silence_frames = 0
            
            # If we've had enough consecutive silence, break the loop
            if silence_frames > silence_consecutive_frames:
                break

            # Append the frames
            frames.append(data)

    # Stop and close the stream
    print("Done recording.")
    stream.stop_stream()
    stream.close()

    # Terminate the PortAudio interface
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio():
    # Note: you need to be using OpenAI Python v0.27.0 for the code below to work
    audio_file= open(WAVE_OUTPUT_FILENAME, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    return transcript.get('text')

def get_user_input():
    setEyesToGreen()
    record_user()

    setEyesToRed()
    user_input = transcribe_audio()

    print(f"User said: {user_input}")

    return user_input

def setEyesToRed():
    ohbot.setEyeColour(0,10,0)

def setEyesToGreen():
    ohbot.setEyeColour(10,0,0)

def setEyesToWhite():
    ohbot.setEyeColour(10,10,10)

def main():
    ohbot.reset()
    setEyesToWhite()

    # read credentials from file
    with open("credentials.json", "r") as f:
        credentials = json.load(f)

    # set openai api key
    openai.api_key = credentials['openai_key']

    stopf = start_flashing_eyes()
    greeting = generate_greeting()
    stopf()

    speak(greeting)

    user_input = get_user_input()
    add_utterance(user_input, False)
    # ohbot.ejo_play_silence()

    try:
        while not user_input.lower() in ["quit", "exit", "bye", "goodbye", "quit.", "exit.", "bye.", "goodbye."]:
            stopf = start_flashing_eyes()
            bot_response = generate_response()
            stopf()
            speak(bot_response)

            user_input = get_user_input()
            add_utterance(user_input, False)
            # ohbot.ejo_play_silence()
    except Exception as e:
        print (e)
        speak("My brain hurts. I had the following error: " + str(e))
        clear_utterances()
    finally:
        stopf = start_flashing_eyes()
        valediction = generate_valediction()
        stopf()
        speak(valediction)
        ohbot.reset()
        ohbot.close()

if __name__ == "__main__":
    main()