from elevenlabs import generate, play, stream, set_api_key
# from dotenv import load_dotenv
import openai
import subprocess
import sys
from scripts.play import play_audio_file
import toml
from scripts.hashing import quick_hash

with open('config.toml') as f:
    config = toml.load(f)
dotenv_path = config["dotenv_path"]

text_to_speech_source = config["text_to_speech_source"]

# load_dotenv(dotenv_path)
import os 
api_key = os.getenv("ELEVEN_API_KEY")
# print(api_key)


# only do this if you want the generation to be linked to an account
# set_api_key(api_key=api_key)


def synth_static(text, voice=config["voice"]):
    voice = voice
    audio = generate(
        text=text,#f"Hi! My name is {voice}, nice to meet you!",
        voice=voice
        # api_key=api_key
    )
    # file_path = "/dictation/speech_synthesis.mp3"
    # with open(file_path, 'wb') as file:
    #     file.write(audio)

    # return audio
    play(audio)

def say(text, voice=config["voice"], cache_audio=True):  # streaming
    # assert text_to_speech_source == "elevenlabs"
    if text_to_speech_source == "elevenlabs":
        audio_stream = generate(
            text=text,
            stream=True,
            voice=voice
        )

        stream(audio_stream)
    elif text_to_speech_source == "openai":
        # print(f"{text = }")
        text_hash = quick_hash(text)
        print(f"{text_hash = }")
        filepath = f"audio-cache/{text_hash}.mp3"
        if os.path.isfile(filepath):
            print("cache hit")
            play_audio_file(filepath)
        else:
            print("cache miss")
            client = openai.OpenAI()

            print("making audio request...")
            response = client.audio.speech.create(
                model=config["voice_model"],
                voice=voice,
                input=text,
            )
            # process = subprocess.Popen(['mpg123', '-'], stdin=subprocess.PIPE)
            process = subprocess.Popen(['mpv','--no-terminal', '-'], stdin=subprocess.PIPE)
            for chunk in response.iter_bytes(chunk_size=4096):
                if chunk:
                    process.stdin.write(chunk)
                    process.stdin.flush()
            process.stdin.close()
            process.wait()


            if cache_audio:
                print(f"saving {filepath} to cache...")
                response.stream_to_file(filepath)


def play_every_voice(sample):
    for voice in voices():
        print(f"this is {voice.name}...")
        say(sample, voice=voice.name)
        print("\n")

if __name__ == "__main__":
    from elevenlabs import voices
    print('start...')
    # sample = "Our Father, which art in heaven, Hallowed be thy Name. Give us this day our daily bread. And forgive us our trespasses, As we forgive them that trespass against us."
    sample = "audio test"
    # play_every_voice(sample)
    say(sample, "Daniel")