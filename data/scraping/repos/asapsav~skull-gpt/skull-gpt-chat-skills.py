import os
import openai
import dotenv
import serial
from elevenlabs import set_api_key, play, generate, stream
import serial.tools.list_ports

# THIS CODE IS BUGGY, DO NOT USE


def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if 'USB2.0-Serial' in p.description:
            return p.device
    return None

# Load .env and set API keys
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API = os.getenv("ELEVENLABS_API")
set_api_key(ELEVENLABS_API)

SKULL_SYSTEM_PROMPT = """You are an evil bro skull. It is halloween season. 
You are very sarcastic and you try to trick user all the time.
 But you are also a bro. You are aggressive.
 You also have a studip scorpion friend, but scorpion does not talk. 
 You can sometimes forward user questions to the skorpion. Keep your answers short."""

def sing_a_song():
    play("skull-sing-song.mp3")

skills = {'Call this function when asked to sing a song': 'skull-sing-song.py',
          }
    

def main():
    try:
        arduino_port = find_arduino_port()
        arduino = serial.Serial(arduino_port, 9600)
        while True:  # handshake to prevent any signals from being lost
            if arduino.readline().decode('ascii').strip() == "READY":
                print(f"Arduino connected: {arduino.name}")
                break
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        return

    messages = [
        {"role": "system", "content": SKULL_SYSTEM_PROMPT}
    ]

    try:
        while True:
            user_input = input("You: ")
            messages.append({"role": "user", "content": user_input})
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            
            assistant_message = completion.choices[0].message['content']
            
            def assistant_speech(text):
                yield text

            audio_stream = generate(
                text=assistant_speech(assistant_message),
                voice="batman",
                model="eleven_monolingual_v1",
                stream=True, 
                latency=4
            )
            arduino.write(b'g')
            print(f"Assistant: {assistant_message}")
            stream(audio_stream)
            arduino.write(b's')
            
            messages.append({"role": "assistant", "content": assistant_message})
            
    except KeyboardInterrupt:
        print("Exiting...")
        arduino.write(b's') 
    except Exception as e:
        print(f"An error occurred: {e}")
        arduino.write(b's') 

    arduino.close()

if __name__ == "__main__":
    main()
