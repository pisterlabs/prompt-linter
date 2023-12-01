from elevenlabslib import *
from colorama import Fore, Back, Style
import openai
import pyaudio
import wave
import os
import keyboard
import importlib
import sys
import subprocess


#ENTER OPENAI KEY HERE 

openai.api_key = "YOUR_API_KEY"

#ENTER ELEVENLABS KEY HERE

user = ElevenLabsUser("YOUR_API_KEY")



required_packages = [
    'openai',
    'keyboard',
    'pyaudio',
    'colorama',
    'wave',
    'elevenlabslib',
]

#define parmeters of the chatbot

INSTRUCTIONS = """You are ChatGPT, playing the role of Voicy, a vocal smart assitant that is always ready to help by providing accurate, short and usefull anwsers. Always try to give anwser of a reasonable lenght, if possible make it short, your focus is efficenty"""

TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 10

#audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
transcript = ""

# Get the current working directory (repository root)
repo_root = os.path.dirname(os.path.abspath(__file__))

# Set the output folder and file name
output_folder = "output_audios"
output_file = "output.wav"

# Create the full path to the output file
output_file_path = os.path.join(repo_root, output_folder, output_file)

# Assign the output file path to WAVE_OUTPUT_FILENAME
WAVE_OUTPUT_FILENAME = output_file_path

# Create an instance of PyAudio
audio = pyaudio.PyAudio()


# Define a global variable to store the audio frames
frames = []

response = ""




def get_response(instructions, previous_questions_and_answers, new_question):
    """Get a response from ChatCompletion
    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        previous_questions_and_answers: Chat history
        new_question: The new question to ask the bot
    Returns:
        The response text
    """
    # build the messages
    messages = [
        { "role": "system", "content": instructions },
    ]
    # add the previous questions and answers
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": new_question })

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content

def install_missing_packages(packages):
    command = [sys.executable, "-m", "pip", "install", *packages]
    try:
        subprocess.check_call(command)
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

missing_packages = []
for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        missing_packages.append(package)
if missing_packages:
    print(f"The following packages are missing: {', '.join(missing_packages)}")
    print("Attempting to install them automatically...")
    install_missing_packages(missing_packages)


# Define a callback function to read audio data as it becomes available
def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
    return (in_data, pyaudio.paContinue)
    

def record_audio():
    # Wait for the user to start the recording
    print("Waiting for Ctrl+Shift+C to start recording...")

    # Wait for the shortcut to start recording
    keyboard.wait('ctrl+shift+c')

    # Open a new audio stream for recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=None,
                        output_device_index=None,
                        stream_callback=callback)

    # Start the audio stream
    stream.start_stream()

    print("Recording... Press Ctrl+Shift+X to stop recording.")
    
    # Record the audio until the user stops it using the shortcut
    keyboard.wait('ctrl+shift+x')

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    

    # Save the recorded audio to a WAV file
    with wave.open(WAVE_OUTPUT_FILENAME, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    frames.clear()  # clear the frames list after saving the audio

    # Close the stream after the first recording
    if stream is not None:
        stream.stop_stream()
        stream.close()
        stream = None  # reset the stream variable


def transcribe_audio():
    global transcript
    audio_file= open(WAVE_OUTPUT_FILENAME, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)



def main():
    # keep track of previous questions and answers
    previous_questions_and_answers = []
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("beggining of loop")
        record_audio()
        transcribe_audio()
        
        # ask the user for their question
        new_question = str(transcript)
        response = get_response(INSTRUCTIONS, previous_questions_and_answers, new_question)

        print(response)

        # add the new question and answer to the list of previous questions and answers
        previous_questions_and_answers.append((new_question, response))

        # print the response
        print(Fore.CYAN + Style.BRIGHT + "AI:" + Style.NORMAL + response)
        voice = user.get_voices_by_name("Rachel")[0]  # This is a list because multiple voices can have the same name
        voice.generate_and_play_audio(str(response), playInBackground=False)
        for historyItem in user.get_history_items():
            if historyItem.text == "Test.":
                # The first items are the newest, so we can stop as soon as we find one.
                historyItem.delete()
                break
        print("end of looop")

if __name__ == "__main__":
    main()