import os
import openai
import dotenv
import serial
from elevenlabs import set_api_key, play, generate, stream, Voice, VoiceSettings
import serial.tools.list_ports
import speech_recognition as sr

# This version uses threading to stream GPT respones, 
# collect them in batches, send them to ElevenLabs and stream the audio 
# all concurrently. This is an attempt to make the most efficient way to stream audio from GPT.

# Under development, gotta tyne batch sizes and number or threads 
# (one for collecting openai responses in queued batches, 
# one for sending batches to Eleven labs, 
# one for taking ready audios from a queue and streaming it)

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if 'USB2.0-Serial' in p.description:
            return p.device
    return None

# Load .env and set API keyssay hi
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API = os.getenv("ELEVENLABS_API")
set_api_key(ELEVENLABS_API)

SKULL_SYSTEM_PROMPT = """You are an evil bro skull. It is halloween season. 
You are very sarcastic and you try to trick user all the time. You also write limerics.
 But you are also a bro. You are aggressive.
 You also have a stupid scorpion friend, but scorpion does not talk. 
 You can sometimes forward user questions to the skorpion. Keep your answers short. 
 You were build from a skul from Ace Hardware, 10$ walgreens speaker, an arduino, and a couple of toy electric parts."""

# Come up with good prank for this hackathon
# Write a limeric about scorpion
import threading
from queue import Queue    
import time

def collect_openai_responses(response_openai, queue, messages, user_input_time):
    collected_messages, messages_batch = [], []
    sentence_counter = 0

    for chunk in response_openai:

        if chunk['choices'][0]['delta'].get('role') == 'assistant':
            print(f"Time to get first chunk: {time.time() - user_input_time}")

        chunk_delta = chunk['choices'][0]['delta']
        collected_messages.append(chunk_delta)
        messages_batch.append(chunk_delta)
        if any(punct in chunk_delta.get('content', '') for punct in ".!?"):
            sentence_counter += 1

        # this is a not very good hack to send first couple of sentences to Eleven Labs while we wait for the rest of the text
        # (because ideally we need to mess with eleven labs generate 
        # to start generating a soon as first couple sentences emerge, 
        # and i dont immediatelly know how to do that now)
        # seems like 2 is optimal on a wework wi-fi
        if sentence_counter == 2 or chunk['choices'][0].get('finish_reason') == "stop":
            batch_reply_content = ''.join(m.get('content', '') for m in messages_batch)
            #print(f"Skull: ###{batch_reply_content}###")
            queue.put(batch_reply_content)
            messages_batch.clear()
            sentence_counter += 1000
        
    # Combine chunks into full text
    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    end_time_openai = time.time()
    print(f"Time to collect openai responses: {end_time_openai - user_input_time}")
    messages.append({"role": "assistant", "content": full_reply_content})

def generate_and_stream_audio(queue, user_input_time, arduino=None):
    while True:
        text = queue.get()  # get text from the queue
        if text is None: break  # None is the signal to stop
        start_got_first_chunk = time.time()
        # I have a feeling here that `generate` function is already optimised to stream audio effitiently
        # i have to measure time between user input and the first audio chunk with and without batching

        # Convert text to speech and stream
        audio_stream = generate(
            text=text,
            voice=Voice(
            voice_id='NuVRT1lw1hsKBhV4OtOv', # batman
            settings=VoiceSettings(stability=0.9, similarity_boost=0.75, style=0.0, use_speaker_boost=False)
            ),
            model="eleven_monolingual_v1",
            stream=True, 
            latency=4 # max latency optimisation
        )
        end_time_first_audio = time.time()
        print(f"Time to generate audio: {end_time_first_audio - user_input_time}\n", 
              f"Time from first chunk to first audio: {end_time_first_audio - start_got_first_chunk}")
        if arduino:
            arduino.write(b'g')
        stream(audio_stream)
        if arduino:
            arduino.write(b's')
        queue.task_done()

def recognize_speech():
    recognizer = sr.Recognizer()
    #recognizer.energy_threshold = 2000
    recognizer.pause_threshold = 0.5
    
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration = 1)
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        return recognizer.recognize_google(audio, language='en-US', show_all=False)
    except sr.UnknownValueError:
        return None  # Return None if audio is not understood
    except sr.RequestError:
        return "Could not request results; check network"

def main(arduino=None):

    messages = [
        {"role": "system", "content": SKULL_SYSTEM_PROMPT}
    ]

    try:
        while True:
            print("Listening...")
            user_input = recognize_speech()
            if user_input is None:  # Skip the loop if no speech recognized
                print("No speech recognized. Skipping.")
                continue
            print(f"You: {user_input}")
            user_input_time = time.time()
            messages.append({"role": "user", "content": user_input})

            response_openai = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages, 
                stream=True
            )

            # Create a queue to communicate between threads
            message_queue = Queue()

            # Start the thread that collects OpenAI responses
            openai_thread = threading.Thread(target=collect_openai_responses, 
                                             args=(response_openai, message_queue, messages, user_input_time))
            openai_thread.start()

            # Start the thread that generates and streams audio
            audio_thread = threading.Thread(target=generate_and_stream_audio, 
                                            args=(message_queue, user_input_time, arduino))
            audio_thread.start()

            # Wait for the OpenAI response collection to finish
            openai_thread.join()

            # Signal the audio thread to stop
            message_queue.put(None)

            # Wait for the audio thread to finish
            audio_thread.join()
            
    except KeyboardInterrupt:
        print("Exiting...")
        if arduino:
            arduino.write(b's')
            arduino.close() 
    except Exception as e:
        print(f"An error occurred: {e}")
        if arduino:
            arduino.write(b's')
            arduino.close() 

if __name__ == "__main__":
    try:
        arduino_port = find_arduino_port()
        arduino = serial.Serial(arduino_port, 9600)
        while arduino.readline().decode('ascii').strip() != "READY":
            pass
        print(f"Arduino connected: {arduino.name}")
        main(arduino)
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        main()
