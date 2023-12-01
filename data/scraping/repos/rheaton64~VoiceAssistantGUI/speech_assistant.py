import queue
import pyaudio
import sys
import wave
import shutil
import keyboard
import tempfile
import openai
import json
import os
import itertools
import threading
import time 
from colorama import init, Fore
from langchain.memory import ConversationBufferWindowMemory
from query_analysis import prep_all_inputs
from assistant_agent import load_assistant_agent
from callbacks import AssistantCallbackHandler
from dotenv import load_dotenv
load_dotenv(override=True)
# Parameters for recording
CHUNK = 1024
FORMAT = pyaudio.paInt24
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# Start Colorama
init()



# Global variables for locks and queues
audio_queue = queue.Queue()
audio_display_queue = queue.Queue()
action_queue = queue.Queue()


# Global variables for recording and displaying
frames = []
p = None
stream = None
recording_done = threading.Event()
interrupt_event = threading.Event()
is_running = threading.Event()
is_playing = threading.Event()
is_recording = threading.Event()
action_pending = threading.Event()
display_buffer = []

# Spinner for loading symbol
def spinning_cursor():
    while True:
        for cursor in itertools.cycle(['-', '\\', '|', '/']):
            yield cursor

spinner = spinning_cursor()

def get_terminal_width():
    columns, _ = shutil.get_terminal_size()
    return columns

def is_playing_audio():
    return is_playing.is_set()

def is_recording_audio():
    return is_recording.is_set()

def is_running_llm():
    return is_running.is_set()


def display_spinning_icon():
    while True:
        if audio_display_queue.qsize() > 0:
            display_buffer.append(audio_display_queue.get())
        if is_playing_audio():
            sys.stdout.write(Fore.GREEN + 'Assistant: ' + Fore.RESET + Fore.GREEN + ' Generating... ' + next(spinner))  # Move cursor up one line
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b' * (len('Assistant: ' + Fore.RESET + Fore.GREEN + ' Generating... ') + 1) + Fore.RESET)  # Move cursor down one line after erasing
        if is_recording_audio():
            sys.stdout.write(Fore.LIGHTBLUE_EX + 'Recording... ' + next(spinner))  # Move cursor up one line
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b' * (len('Recording... ') + 1) + Fore.RESET)  # Move cursor down one line after erasing
        else:
            time.sleep(0.1)

def send_to_streamlit_queue():
    pass

def stream_callback(in_data, frame_count, time_info, status):
    global frames
    frames.append(in_data)
    return (in_data, pyaudio.paContinue)



def start_recording():
    global stream, is_recording
    if stream is None or not stream.is_active():
        sys.stdout.write('\r' + ' ' * get_terminal_width() + '\r')
        is_recording.set()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=stream_callback)
        stream.start_stream()
        

def stop_recording():
    global stream, recording_done, is_recording, p
    if stream is not None and stream.is_active():
        is_recording.clear()
        stream.stop_stream()
        stream.close()
        stream = None
        recording_done.set()


def get_voice_input():
    global frames, stream, recording_done, p
    hotkey = '`'
    frames = []
    p = pyaudio.PyAudio()
    # Create a temporary file for the recording
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".wav", dir='./record_outputs')
    WAVE_OUTPUT_FILENAME = temp_file.name  # Use the temporary file's name

    sys.stdout.write('\r' + ' ' * get_terminal_width() + '\r')
    sys.stdout.flush()

    # Open a new stream for recording
    print()
    sys.stdout.write(Fore.LIGHTBLUE_EX + 'Press the key to start recording and release to stop recording' + Fore.RESET)
    sys.stdout.flush()
   
    # Register key press and release events
    keyboard.add_hotkey(hotkey, start_recording, suppress=True)
    keyboard.add_hotkey(hotkey, stop_recording, trigger_on_release=True, suppress=True)
    # keyboard.add_hotkey(hotkey, lambda: set_interrupted(True), suppress=True)

    # Wait for the recording to be done
    recording_done.wait()
    recording_done.clear()

    # Unregister key press and release events
    keyboard.remove_hotkey(start_recording)
    p.terminate()

    sys.stdout.flush()

    temp_file.close()
    # Save the recorded audio to a file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    audio_file= open(WAVE_OUTPUT_FILENAME, "rb") 

    transcript = openai.Audio.transcribe("whisper-1", audio_file, api_key=os.getenv('OPENAI_API_KEY'))
    
    
    return transcript.text

memory=ConversationBufferWindowMemory(return_messages=True)

agent_chat_chain = load_assistant_agent(memory, AssistantCallbackHandler(voice='BBC', api_key=os.getenv('XILABS_API_KEY'), running_event=is_running, playing_event=is_playing, action_pending=action_pending, action_queue=action_queue, display_queue=audio_display_queue))

def save_context_to_file(context, file_path='context.json'):
    with open(file_path, 'w') as f:
        json.dump(context, f)

def load_context_from_file(file_path='context.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            context = json.load(f)
    else:
        context = {'tools': []}
    return context

# Start the spinning icon display thread
threading.Thread(target=display_spinning_icon, daemon=True).start()

action_response_template = """<SYSTEM>
    Action Response:
    {action_response}

    Notes from the System:
    The above is the response from the action executor. You don't necessarily need to repeat it back to the user verbatim, but you should use it to inform your response to the user.
    The system doesn't remember information from previous actions, so if you use a follow-up action, you'll need to repeat any information you want to use from the previous action.
    Be mindful that your response is being fed into a text-to-speech engine, so you may want to avoid using links or other text that may not sound good when read aloud.
</SYSTEM>"""

def get_action_output(context):
    while True:
        if action_queue.empty():
            time.sleep(0.2)
        else:
            res = action_queue.get() # dict with 'response'
            res_msg = action_response_template.format(action_response=res['response'])
            context['tools'].append(res)
            print(Fore.RED +res_msg+ Fore.RESET)
            print()
            return res_msg
            

context = {}

def start_voice_input():
    state = None
    while True:
        context = load_context_from_file()
        if not action_pending.is_set() and action_queue.empty():
            transcript = get_voice_input()
            (prepped_input, edits, state) = prep_all_inputs(transcript, memory, context, state)
            edits_str = "|".join(k for k in edits.keys() if edits[k] == True)
            print(Fore.CYAN + "Human:" + Fore.RESET, transcript, Fore.RED + "Edits:" + Fore.RESET, edits_str)
            print()
        else:
            prepped_input = get_action_output(context)
        response = agent_chat_chain.run(prepped_input)
        while True:
            if is_playing_audio():
                time.sleep(0.1)
            else:
                break
        sys.stdout.write('\r' + ' ' * get_terminal_width() + '\r')
        sys.stdout.flush()
        print(Fore.GREEN + "Assistant:" + Fore.RESET, response)
        save_context_to_file(context)

if __name__ == "__main__":
    # flask_thread = threading.Thread(target=start_flask_app)
    voice_input_thread = threading.Thread(target=start_voice_input, daemon=True)
    # flask_thread.start()
    voice_input_thread.start()
    # flask_thread.join()
    voice_input_thread.join()