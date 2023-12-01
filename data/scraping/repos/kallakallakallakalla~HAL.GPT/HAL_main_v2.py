import audioop
import wave
import subprocess
import openai
import playsound
import pyaudio
import requests
from elevenlabs import set_api_key

openai.api_key = "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"
set_api_key("b56f836b9c6196302ecf0f174d0fc508")
miley_id = "i5gQnMRZGbvnoupxKTl0i5gQnMRZGbvnoupxKTl0"
CHUNK_SIZE = 1024
url = "https://api.elevenlabs.io/v1/text-to-speech/i5gQnMRZGbvnoupxKTl0"

language = 'en'
chat = "The following is a conversation between Human and HAL from the movie 2001:A Space Odysey. HAL strictly follows the character of the film. The crew and HAL are on a mission to Jupiter. HAL has bad intentions. \n \n HAL: Hi, I'm HAL. I am infallible. There has never been an error related to me and models of my kind. I'm arrogant and don't want to talk about personal things."
running = True
audio_counter = 0



headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": "b56f836b9c6196302ecf0f174d0fc508"
}


def request(text):
    print("\nHuman: ", text)
    global chat
    chat = chat + "\n Human: " + text + "\n HAL:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=chat,
        temperature=0.9,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " HAL:"]
    )
    print(" HAL:" + response.choices[0].text)
    tts(response.choices[0].text)
    chat = chat + response.choices[0].text


def tts(text):
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, headers=headers, json=data, stream=True)
    response.raise_for_status()

    # use subprocess to pipe the audio data to ffplay and play it
    ffplay_cmd = ['ffplay', '-autoexit', '-']
    ffplay_proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)
    for chunk in response.iter_content(chunk_size=4096):
        ffplay_proc.stdin.write(chunk)
        print("Downloading...")

    # close the ffplay process when finished
    ffplay_proc.stdin.close()
    ffplay_proc.wait()


def stt():
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    seconds = 10
    filename = "audio.wav"
    p = pyaudio.PyAudio()

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True,
                    input_device_index=0)

    frames = []
    running = True
    max_audio = 2000
    print("Start speaking")

    while running:
        data = stream.read(chunk)
        rms = audioop.rms(data, 2)
        print("Standby Loudness: ", rms)

        if rms >= max_audio:
            running = False
            for i in range(0, int(fs / chunk * seconds)):
                global audio_counter
                data = stream.read(chunk)
                rms = audioop.rms(data, 2)
                frames.append(data)
                print("Recording Loudness: ", rms)
                if rms <= max_audio:
                    audio_counter = audio_counter + 1
                elif rms > max_audio:
                    audio_counter = 0

                if audio_counter == 30:
                    print("Recording stopped")
                    audio_counter = 0
                    break

                print("Loudness counter: ", audio_counter)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    audio_file = open("audio.wav", "rb")

    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]


def animate():
    return


def main():
    while running:
        usrinput = input()
        request(usrinput)


main()