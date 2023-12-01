import whisper
import playsound  # to play saved mp3 file
from gtts import gTTS  # google text to speech
import os  # to save/open files
import openai
import requests
import pyaudio
import wave
num = 1
openai.api_key = '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'
def assistant_speaks(output):
    global num
    num += 1
    print("Proxie : ", output)
    toSpeak = gTTS(text=output, lang='en', slow=False)
    file = str(num) + ".mp3"
    toSpeak.save(file)
    playsound.playsound(file, True)
    os.remove(file)
def get_audio():
    chunk = 1024


    sample_format = pyaudio.paInt16
    chanels = 2

    smpl_rt = 44400
    seconds = 4
    filename = "prox.wav"


    pa = pyaudio.PyAudio()

    stream = pa.open(format=sample_format, channels=chanels,
                     rate=smpl_rt, input=True,
                     frames_per_buffer=chunk)

    print('Recording...')


    frames = []

    # Store data in chunks for 8 seconds
    for i in range(0, int(smpl_rt / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate - PortAudio interface
    pa.terminate()

    print('Done !!! ')

    # Save the recorded data in a .wav format
    sf = wave.open(filename, 'wb')
    sf.setnchannels(chanels)
    sf.setsampwidth(pa.get_sample_size(sample_format))
    sf.setframerate(smpl_rt)
    sf.writeframes(b''.join(frames))
    sf.close()
    get_audio()
    model = whisper.load_model("base")
    globals()['result'] = model.transcribe("prox.wav")
    globals()['text'] = result["text"]
def sarcasm():
    globals()['sa'] = get_audio().lower()
    response = response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Marv is a chatbot that reluctantly answers questions with sarcastic responses:\n\nYou:{str(sa)}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    stop = ["\n", " Human:", ]
    answer = response.choices[0].text.strip()
    assistant_speaks(answer)
def grammar():
    get_audio()
    response = response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Correct this to standard English:\n\n{str(g)}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    stop = ["\n", " Human:", ]
    answer = response.choices[0].text.strip()
    assistant_speaks(answer)
def qanda():
    get_audio()
    response = response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Q: {str(q)}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    stop = ["\n", " Human:", ]

    answer = response.choices[0].text.strip()
    assistant_speaks(answer)
def summarise():
    get_audio()
    response = response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Summarize this for a second-grade student:\n\n{str(s)}",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    stop = ["\n", " Human:", ]
    answer = response.choices[0].text.strip()
    assistant_speaks(answer)