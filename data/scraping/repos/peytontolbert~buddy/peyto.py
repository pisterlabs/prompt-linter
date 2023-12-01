# Necessary Libraries
import sounddevice as sd
import wavio as wv
import datetime
import json
import whisper
import openai
import os
import sys
import re
from nltk.tokenize import sent_tokenize
import nltk
import requests
from TTSTT2.hparams import create_hparams
from TTSTT2.model import Tacotron2
from TTSTT2.layers import TacotronSTFT
from hifigan.audio_processing import griffin_lim
from hifigan.denoiser import Denoiser
from hifigan.env import AttrDict
from hifigan.meldataset import mel_spectrogram, MAX_WAV_VALUE
from hifigan.models import Generator
import resampy
import matplotlib.pyplot as plt
import scipy.signal
from text import text_to_sequence
from pydub import AudioSegment
#from pydub.playback import play
from dotenv import load_dotenv
import time
import numpy as np
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
nltk.download('punkt')
print(torch.version.cuda)
# Constants and Configuration
load_dotenv()
FREQ = 44100  # Sample frequency
DURATION = 5  # Duration of recording in seconds
MODEL = whisper.load_model("base")  # Load Whisper model
OPENAI_API_KEY = os.getenv('OPENAI_KEY')  # Replace with your OpenAI API key
openai.api_key = OPENAI_API_KEY  # Set OpenAI API key
# Data Initialization
pronounciation_dictionary = False #@param {type:"boolean"}

parsedname, parsednumber = False, False
superres_strength = 10 #@param {type:"number"}
thisdict = {}
for line in reversed((open('merged.dict.txt', "r").read()).splitlines()):
    thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()

retrievedinformation = [{"name": "", "email": "", "phone": ""}]
messages = [
    {"role": "system", "content": """
You are Peyton, and have been working on an autonomous agent. You are on a call with Eric discussing your invention. 

[EXAMPLE] 
Hey Eric, this is Peyton. I am totally not a robot, and Peyton. I got my tattoo yesterday

"""}
]


salespitchprompt = """
Act as Peyton and give short responses. continue the conversation about your week, all you've been doing is building and working. You're also playing league of legends right now, continue the conversation
[REMEMBER] Keep responses short, only 1-2 sentences!
"""

closeoutprompt = """
Try to h
"""

class ChatGPT:
    def __init__(self):
        pass
    @staticmethod
    def chat_with_gpt3(messages):
        retries = 20
        delay = 5
        for i in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages=messages,
                    temperature=0.9
                )
                print(response)
                return response['choices'][0]['message']['content']
            except openai.error.ServiceUnavailableError:
                if i < retries - 1:
                    time.sleep(delay)
                else:
                    raise


TACOTRON_MODEL = "Peyton-100"
HIFIGAN_MODEL = "config_v1.json"

# Load Tacotron2
def load_tacotron(model_name):
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    model = Tacotron2(hparams)
    checkpoint_path = os.path.join("models", model_name)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model = model.cuda().eval().half()
    return model, hparams

# Load HiFi-GAN
def load_hifigan(MODEL_ID, conf_name):
    conf = os.path.join("hifigan", conf_name)
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device("cuda"))
    # Assuming your HiFi-GAN model is named similarly to your Tacotron2 model but ends with '_hifigan'
    if MODEL_ID == 1:
        hifigan_model_path = os.path.join("models", "Superres_Twilight_33000")
    else:
        hifigan_model_path = os.path.join("models", "g_02500000")
    state_dict_g = torch.load(hifigan_model_path, map_location=torch.device("cuda"))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser

tacotron, tacotron_hparams = load_tacotron(TACOTRON_MODEL)
hifigan, h, denoiser = load_hifigan("universal", HIFIGAN_MODEL) 
hifigan_sr, h2, denoiser_sr = load_hifigan(1, "config_32k.json")


max_duration = 35 #@param {type:"integer"}
tacotron.decoder.max_decoder_steps = max_duration * 80
stop_threshold = .8 #@param {type:"number"}
tacotron.decoder.gate_threshold = stop_threshold

def ARPA(text, punctuation=r"!?,.;", EOS_Token=True):
    out = ''
    for word_ in text.split(" "):
        word=word_; end_chars = ''
        while any(elem in word for elem in punctuation) and len(word) > 1:
            if word[-1] in punctuation: end_chars = word[-1] + end_chars; word = word[:-1]
            else: break
        try:
            word_arpa = thisdict[word.upper()]
            word = "{" + str(word_arpa) + "}"
        except KeyError: pass
        out = (out + " " + word + end_chars).strip()
    if EOS_Token and out[-1] != ";": out += ";"
    return out


def synthesize_audio(text, pronounciation_dictionary):
    for i in [x for x in text.split("\n") if len(x)]:
        if not pronounciation_dictionary:
            if i[-1] != ";": i=i+";"
        else: i = ARPA(i)
        with torch.no_grad(): # save VRAM by not including gradients
            sequence = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, _, alignments = tacotron.inference(sequence)
            
            #if show_graphs:
                #plot_data((mel_outputs_postnet.float().data.cpu().numpy()[0],
                        #alignments.float().data.cpu().numpy()[0].T))
            # Plotting the mel-spectrogram
            plt.imshow(mel_outputs_postnet[0].cpu().detach().numpy(), origin="lower", aspect="auto")
            plt.title("Mel-Spectrogram")
            plt.colorbar()
            plt.show()
            y_g_hat = hifigan(mel_outputs_postnet.float())
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
            # Resample to 32k
            audio_denoised = audio_denoised.cpu().numpy().reshape(-1)
            normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
            audio_denoised = audio_denoised * normalize
            wave = resampy.resample(
                audio_denoised,
                h.sampling_rate,
                h2.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            # HiFi-GAN super-resolution
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device("cuda"))
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                h2.n_fft,
                h2.num_mels,
                h2.sampling_rate,
                h2.hop_size,
                h2.win_size,
                h2.fmin,
                h2.fmax,
            )
            y_g_hat2 = hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze()
            audio2 = audio2 * MAX_WAV_VALUE
            audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0]

            # High-pass filter, mixing and denormalizing
            audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
            b = scipy.signal.firwin(
                101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
            )
            y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
            y *= superres_strength
            y_out = y.astype(np.int16)
            y_padded = np.zeros(wave_out.shape)
            y_padded[: y_out.shape[0]] = y_out
            sr_mix = wave_out + y_padded
            sr_mix = sr_mix / normalize
            sd.play(sr_mix.astype(np.int16), samplerate=h2.sampling_rate)
            sd.wait()  # Wait until audio playback is done

def text_to_speech(text):
    synthesize_audio(text, pronounciation_dictionary)

def record_audio(silence_threshold=0.05, silence_duration=1.0, min_duration=3.0):
    print('Recording')
    ts = datetime.datetime.now()
    filename = ts.strftime("%Y-%m-%d_%H-%M-%S")  # Changed ':' to '_'
    filepath = f"./recordings/{filename}.wav"
    min_chunks = int(FREQ * min_duration / DURATION)  # Minimum number of chunks to record

    # Continuous recording function
    with sd.InputStream(samplerate=FREQ, channels=1) as stream:
        audio_frames = []
        silent_chunks = 0
        silence_chunk_duration = int(FREQ * silence_duration / DURATION)  # Number of chunks of silence before stopping

        has_input = False  # Flag to check if there's any non-silent input
        total_chunks = 0  # Counter for total chunks recorded


        while True:
            audio_chunk, overflowed = stream.read(DURATION)
            audio_frames.append(audio_chunk)

            # Check volume of the audio chunk
            volume_norm = np.linalg.norm(audio_chunk) / len(audio_chunk)
            
            # If volume below the threshold, we consider it as silence
            if volume_norm < silence_threshold:
                if has_input:  # Only increment silent_chunks if we've had non-silent input
                    silent_chunks += 1
            else:
                silent_chunks = 0
                has_input = True  # Set the flag when we detect non-silent input
            total_chunks+=1

            # If silence for a certain duration after non-silent input, stop recording
            if silent_chunks > silence_chunk_duration and has_input and total_chunks > min_chunks:
                break

        # Save the audio
        recording = np.concatenate(audio_frames, axis=0)
        wv.write(filepath, recording, FREQ, sampwidth=2)

    return filename

def transcribe_audio(filename):
    audio = whisper.load_audio(f"./recordings/{filename}.wav")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(MODEL.device)
    options = whisper.DecodingOptions(language='en', fp16=False)
    result = whisper.decode(MODEL, mel, options)
    if result.no_speech_prob < 0.5:
        return result.text
    else:
        return None

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format='wav')

def sales_pitch(transcription):
    messages.append({"role": "user", "content": transcription})
    messages.append({"role": "assistant", "content": salespitchprompt})
    response = ChatGPT.chat_with_gpt3(messages)
    messages.append({"role": "assistant", "content": response})
    return response

def first_message():
    response = ChatGPT.chat_with_gpt3(messages)
    messages.append({"role": "assistant", "content": response})
    return response

def close_call(transcription):
    messages.append({"role": "user", "content": transcription})
    messages.append({"role": "system", "content": closeoutprompt})
    response = ChatGPT.chat_with_gpt3(messages)
    messages.append({"role": "assistant", "content": response})
    return response

def parse_client_details(transcription):
    client_details = {
        "name": parse_name(transcription),
        "number": parse_number(transcription),
    }
    return client_details
def parse_name(transcription):
    nameprompt="""You are a name parser.
    Your job is to read the user response and parse a name if included in the text. 
    If a name is not found type "no name".
    Only reply with one name. Follow the example below
    
    [EXAMPLE]
    user: Sure, Yeah go ahead
    [RESPONSE]
    assistant: no name
    [EXIT]
    
    [EXAMPLE 2]
    user: Yeah my name is Peyton
    [RESPONSE]
    assistant: Peyton
    [EXIT]
    """
    gptmessages=[{"role": "system", "content": nameprompt}, {"role": "user", "content": transcription}]
    name = ChatGPT.chat_with_gpt3(gptmessages)
    if name!="no name":
        parsedname=True
        return name
    else:
        return ""


def parse_number(transcription):
    numberprompt="""You are a number parser.
    Your job is to read the user response and parse a number if included in the text. 
    If a number is not found type "no number".
    Only reply with one number. Follow the example below

    [EXAMPLE]
    user: Sure, Yeah go ahead
    [RESPONSE]
    assistant: no number
    [EXIT]
    
    [EXAMPLE 2]
    user: Yeah my number is 1234567
    [RESPONSE]
    assistant: 1234567
    [EXIT]
    """
    gptmessages=[{"role": "system", "content": numberprompt}, {"role": "user", "content": transcription}]
    name = ChatGPT.chat_with_gpt3(gptmessages)
    print(name)
    if name!="no number":
        parsednumber=True
        return name
    else:
        return ""


def extract_number(transcription):
    # Mapping words to digits
    word_to_digit = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9"
    }

    # Splitting the transcription into words
    words = transcription.lower().split()
    # Converting word numbers to digits
    digits = [word_to_digit[word] for word in words if word in word_to_digit]
    # Joining the digits to form the number
    number = "".join(digits)

    return number


# Setup and Preparation
def setup():
    if not os.path.exists('./recordings'):
        os.makedirs('./recordings')

# Introduction
def introduction():
    script = first_message()
    text_to_speech(script)
    #convert_mp3_to_wav("response.mp3", "response.wav")
    #play_audio("response.wav")
    filename = record_audio()
    transcription = transcribe_audio(filename)
    return transcription

# Sales Pitch
def sales_pitch_section(transcription):
    response = sales_pitch(transcription)
    text_to_speech(response)
    filename = record_audio()
    transcription = transcribe_audio(filename)
    return transcription

# Update Parsed Info
def update_parsedinfo(parsedinfo, client_details):
    # If the name is not yet filled and a name is found in the transcription
    if not parsedinfo["name"] and client_details["name"]:
        parsedinfo["name"] = client_details["name"]
    # If the name is filled and the number is found in the transcription
    elif parsedinfo["name"] and client_details["number"]:
        parsedinfo["number"] = client_details["number"]
    return parsedinfo
# Ongoing Interaction with Client
def ongoing_interaction(transcription):
    counter = 0
    parsedinfo = {"name": "", "number": ""}
    while True:
        client_details = parse_client_details(transcription)
        if parsedname==True and parsednumber==True:
            send_farewell(client_details)
            exit()
        update_parsedinfo(parsedinfo, client_details)
        print("Parsed Info:", parsedinfo)
        response = close_call(transcription)
        print(response)
        response_wav_path = f"response_{counter}.wav"
        text_to_speech(response)
        filename = record_audio()
        transcription = transcribe_audio(filename)
        counter += 1

# Closing the Interaction
def send_farewell(client_details):
    farewell = "Thank you for using our service. We will contact you shortly. Have a good day!"
    text_to_speech(farewell)
    print(f"Farewell sent. Closing application on customer: {client_details}")

# Main Function
def main():
    setup()
    transcription = introduction()
    response = sales_pitch_section(transcription)
    ongoing_interaction(response)

if __name__ == "__main__":
    main()