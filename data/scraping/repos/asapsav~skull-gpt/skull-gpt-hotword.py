import numpy as np
import pyaudio
import pvporcupine
import os
import dotenv
import openai
import serial
from elevenlabs import set_api_key, generate, stream
import speech_recognition as sr
import serial.tools.list_ports
from pvrecorder import PvRecorder


# THIS CODE KEEP GIVING ME ERROR:


# ... (other functions like find_arduino_port and recognize_speech remain the same)
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if 'USB2.0-Serial' in p.description:
            return p.device
    return None

def recognize_speech(recognizer):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None  # Return None if audio is not understood
    except sr.RequestError:
        return "Could not request results; check network"



dotenv.load_dotenv()
pico_key = os.getenv("PICO_ACCES_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API = os.getenv("ELEVENLABS_API")
set_api_key(ELEVENLABS_API)

porcupine = pvporcupine.create(
  access_key=pico_key,
  keyword_paths=['picovoice_models/yo-skull_en_mac_v3_0_0.ppn']
)


# SKULL_SYSTEM_PROMPT and main function definition
SKULL_SYSTEM_PROMPT = """You are an evil bro skull. It is halloween season. 
You are very sarcastic and you try to trick user all the time.
 But you are also a bro. You are aggressive.
 You also have a studip scorpion friend, but scorpion does not talk. 
 You can sometimes forward user questions to the skorpion. Keep your answers short."""

def main(arduino=None):

    messages = [
        {"role": "system", "content": SKULL_SYSTEM_PROMPT}
    ]

    recognizer = sr.Recognizer()

    
    
    try:
        while True:
            recorder = PvRecorder(
                frame_length=porcupine.frame_length,)
                #device_index=None)
            recorder.start()
            print('Listening for keyword...')
            pcm = recorder.read()
            result = porcupine.process(pcm)
            
            if result >= 0:
                print('Keyword detected, listening for command...')
                recorder.delete()
                #porcupine.delete()

                user_input = recognize_speech(recognizer)

                if user_input is None:
                    print("No command recognized. Skipping.")
                    continue
                print(f"You: {user_input}")

                messages.append({"role": "user", "content": user_input})

                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                assistant_message = completion.choices[0].message['content']

                def assistant_speech(text):
                    yield text

                response_audio_stream = generate(
                    text=assistant_speech(assistant_message),
                    voice="batman",
                    model="eleven_monolingual_v1",
                    stream=True, 
                    latency=4
                )
                if arduino:
                    arduino.write(b'g')
                stream(response_audio_stream)
                if arduino:
                    arduino.write(b's') 
                messages.append({"role": "assistant", "content": assistant_message})
                
    except KeyboardInterrupt:
        print("Exiting...")
        if arduino:
            arduino.write(b's') 
    except Exception as e:
        print(f"An error occurred: {e}")
        if arduino:
            arduino.write(b's') 
    finally:
        #response_audio_stream.close()
        porcupine.delete()
        if arduino:
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
