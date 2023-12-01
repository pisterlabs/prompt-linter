# Copyright (C) MissingNO123 17 Mar 2023

from io import BytesIO
import sys
import time
import struct
full_start_time = time.perf_counter()
import audioop
from datetime import datetime
from dotenv import load_dotenv
# import whisper
from faster_whisper import WhisperModel
import ffmpeg
import openai
import os
import pyaudio
from pynput.keyboard import Listener
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
import re
# import shutil
import threading
import wave

import options as opts
import texttospeech as ttsutils
import uistuff as ui

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# OPTIONS ####################################################################################################################################
CHUNK_SIZE = 1024           # number of frames read at a time
FORMAT = pyaudio.paInt16    # PCM format (int16)
RATE = 48000                # sample rate in Hz

# Variables ###################################################################################################################################
model = None    # Whisper model object

vb_out = None
vb_in = None

frames = []
lastFrame = None
recording = False
speaking = False
trigger = False
panic = False
silence_timeout_timer = None

key_press_window_timeup = time.time()

# opts.tts_engine = ttsutils.WindowsTTS()
# opts.tts_engine = ttsutils.GoogleCloudTTS()
opts.tts_engine = ttsutils.GoogleTranslateTTS()
# opts.tts_engine = ttsutils.eleven
# opts.tts_engine = ttsutils.TikTokTTS()

# Constants
pyAudio = pyaudio.PyAudio()

speech_on = "Speech On.wav"
speech_off = "Speech Sleep.wav"
speech_mis = "Speech Misrecognition.wav"

ip = "127.0.0.1"  # IP and Ports for VRChat OSC
inPort = 9000
outPort = 9001

opts.LOOP = True

whisper_lock = threading.Lock()


# Functions and Class (singular) ##############################################################################################################
# Loads an audio file, play() will play it through vb aux input
class AudioFile:
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """
        init_audio()
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.p.get_format_from_width(self.wf.getsampwidth()),
            channels=self.wf.getnchannels(),
            rate=self.wf.getframerate(),
            output_device_index=vb_in,
            output=True
        )

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(self.chunk)
        while data != b'':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)
            if panic: break
        self.close()

    def close(self):
        """ Graceful shutdown """
        self.stream.close()
        self.p.terminate()


def verbose_print(text):
    if opts.verbosity:
        print(text)


def play_sound(file):
    """ Plays a sound, waits for it to finish before continuing """
    audio = AudioFile(file)
    audio.play()


def play_sound_threaded(file):
    """ Plays a sound without blocking the main thread """
    audio = AudioFile(file)
    thread = threading.Thread(target=audio.play)
    thread.start()


def save_recorded_frames(frames):
    """ Saves recorded frames to a .wav file and sends it to whisper to transcribe it """
    if opts.soundFeedback:
        play_sound_threaded(speech_off)
    recording = BytesIO()
    wf = wave.open(recording, 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(pyAudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    recording.seek(0)
    whisper_transcribe(recording)


def ffmpeg_for_whisper(file):
    import numpy as np
    start_time = time.perf_counter()
    file.seek(0)
    try:
        out, _ = (
            ffmpeg.input('pipe:', loglevel='quiet', threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg", "-nostdin"], input=file.read(), capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    end_time = time.perf_counter()
    verbose_print(f"--FFMPEG for Whisper took: {end_time - start_time:.3f}s")
    return data


def openai_whisper_transcribe(recording):
    """ Transcribes audio in .wav file to text """
    import whisper
    
    if model is None: return
    vrc_chatbox('✍️ Transcribing...')
    verbose_print('~Transcribing...')
    start_time = time.perf_counter()
    
    audio = ffmpeg_for_whisper(recording)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(prompt=opts.whisper_prompt, language='en')
    result = whisper.decode(model, mel, options)
    end_time = time.perf_counter()
    verbose_print(f"--Transcription took: {end_time - start_time:.3f}s, U: {result.no_speech_prob*100:.1f}%")

    # print the recognized text
    print(f"\n>User: {result.text}")

    # if not speech, dont send to cgpt
    if result.no_speech_prob > 0.5:
        vrc_chatbox('⚠ [unintelligible]')
        if opts.soundFeedback: play_sound_threaded(speech_mis)
        verbose_print(f"U: {result.no_speech_prob*100:.1f}%")
        tts('I didn\'t understand that!', 'en')
        vrc_set_parameter('VoiceRec_End', True)
        vrc_set_parameter('CGPT_Result', True)
        vrc_set_parameter('CGPT_End', True)
    else:
        # otherwise, forward text to ChatGPT
        vrc_set_parameter('VoiceRec_End', True)
        vrc_chatbox('📡 Sending to OpenAI...')
        chatgpt_req(result.text)


def whisper_transcribe(recording):
    """ Transcribes audio in .wav file to text using Faster Whisper """
    if model is None:
        return
    vrc_chatbox('✍️ Transcribing...')
    verbose_print('~Transcribing...')

    with whisper_lock:
        start_time = time.perf_counter()
        # audio = ffmpeg_for_whisper(recording) # This adds 500ms of latency with no apparent benefit 
        # Initialize transcription object on the recording
        segments, info = model.transcribe(
            recording, beam_size=5, initial_prompt=opts.whisper_prompt, no_speech_threshold=0.4, log_prob_threshold=0.8)

        verbose_print(f'lang: {info.language}, {info.language_probability * 100:.1f}%')

        # if not speech, dont bother processing anything  
        if info.language_probability < 0.8 or info.duration <= (opts.SILENCE_TIMEOUT + 0.3):
            vrc_chatbox('⚠ [unintelligible]')
            if opts.soundFeedback:
                play_sound_threaded(speech_mis)
            play_sound('./prebaked_tts/Ididntunderstandthat.wav')
            vrc_set_parameter('VoiceRec_End', True)
            vrc_set_parameter('CGPT_Result', True)
            vrc_set_parameter('CGPT_End', True)
            end_time = time.perf_counter()
            verbose_print(f"--Transcription failed and took: {end_time - start_time:.3f}s")
            return

        # Transcribe and concatenate the text segments
        text = ""
        for segment in segments:
            text += segment.text
        text = text.strip()

        end_time = time.perf_counter()
    verbose_print(f"--Transcription took: {end_time - start_time:.3f}s")

    if text == "":
        print ("\n>User: <Nothing was recognized?>")
        vrc_set_parameter('VoiceRec_End', True)
        vrc_set_parameter('CGPT_Result', True)
        vrc_set_parameter('CGPT_End', True)
        return

    # print the recognized text
    print(f"\n>User: {text}")

    # if keyword detected, send to command handler instead
    if text.lower().startswith("system"):
        command = re.sub(r'[^a-zA-Z0-9]', '', text[text.find(' ') + 1:])
        handle_command(command.lower())
        vrc_set_parameter('VoiceRec_End', True)
        vrc_set_parameter('CGPT_Result', True)
        vrc_set_parameter('CGPT_End', True)
        return

    # Repeat input if parrot mode is on 
    if opts.parrot_mode:
        if opts.chatbox and len(text) > 140:
            cut_up_text(f'{text}')
        else:
            vrc_chatbox(f'{text}')
            tts(text)
        vrc_set_parameter('VoiceRec_End', True)
        vrc_set_parameter('CGPT_Result', True)
        vrc_set_parameter('CGPT_End', True)
    else:
        # otherwise, forward text to ChatGPT
        vrc_set_parameter('VoiceRec_End', True)
        vrc_chatbox('📡 Sending to OpenAI...')
        chatgpt_req(text)


def chatgpt_req(text):
    """ Sends text to OpenAI, gets the response, and puts it into the chatbox """
    if len(opts.message_array) > opts.max_conv_length:  # Trim down chat buffer if it gets too long
        opts.message_array.pop(0)
    # Add user's message to the chat buffer
    opts.message_array.append({"role": "user", "content": text})
    # Init system prompt with date and add it persistently to top of chat buffer
    system_prompt_object = [{"role": "system", "content":
                           opts.system_prompt
                           + f' The current date and time is {datetime.now().strftime("%A %B %d %Y, %I:%M:%S %p")} Eastern Standard Time.'
                           + f' You are using {opts.gpt} from OpenAI.'}]
    # create object with system prompt and chat history to send to OpenAI for generation
    message_plus_system = system_prompt_object + opts.message_array
    err = None
    try:
        start_time = time.perf_counter()
        completion = openai.ChatCompletion.create(
            model=opts.gpt.lower(),
            messages=message_plus_system,
            max_tokens=opts.max_tokens,
            temperature=0.5,
            frequency_penalty=0.2,
            presence_penalty=0.5,
            logit_bias={'1722': -100, '292': -100, '281': -100, '20185': -100, '9552': -100, '3303': -100, '2746': -100, '19849': -100, '41599': -100, '7926': -100,
            '1058': 1, '18': 1, '299': 5, '3972': 5}
            # 'As', 'as', ' an', 'AI', ' AI', ' language', ' model', 'model', 'sorry', ' sorry', ' :', '3', ' n', 'ya'
            )
        end_time = time.perf_counter()
        verbose_print(f'--OpenAI API took {end_time - start_time:.3f}s')
        result = completion.choices[0].message.content
        opts.message_array.append({"role": "assistant", "content": result})
        print(f"\n>ChatGPT: {result}")
        # tts(ttsutils.filter(result), 'en')
        # tts(result, 'en')
        # vrc_chatbox('🛰 Getting TTS from 11.ai...')
        if opts.chatbox and len(result) > 140:
            cut_up_text(f'🤖{result}')
        else:
            vrc_chatbox(f'🤖{result}')
            tts(result)
    except openai.APIError as e:
        err = e
        print(f"!!Got API error from OpenAI: {e}")
    except openai.InvalidRequestError as e:
        err = e
        print(f"!!Invalid Request: {e}")
    except openai.OpenAIError as e:
        err = e
        print(f"!!Got OpenAI Error from OpenAI: {e}")
    except Exception as e:
        err = e
        print(f"!!Other Exception: {e}")
    finally:
        if err is not None: vrc_chatbox(f'⚠ {err}')
        vrc_set_parameter('CGPT_Result', True)
        vrc_set_parameter('CGPT_End', True)


def cut_up_text(text):
    """ Cuts text into segments of 144 chars that are pushed one by one to VRC Chatbox """
    global speaking
    # Check if text has whitespace or punctuation
    if re.search(r'[\s.,?!]', text):
        # Split the text into segments of up to 144 characters using the regex pattern
        segments = re.findall(
            r'.{1,143}(?<=\S)(?=[,.?!]?\s|$)|\b.{1,143}\b', text)
    else:
        # Split the text into chunks of up to 144 characters using list comprehension
        segments = [text[i:i+143] for i in range(0, len(text), 143)]
    i = 0
    list = []
    for i, segment in enumerate(segments):
        audio = opts.tts_engine.tts(ttsutils.filter(segment))
        if ( i is not len(segments) - 1 ) and ( opts.tts_engine is not ttsutils.eleven ):
            audio = clip_audio_end(audio)
        list.append((segment, audio))
    # and then
    speaking = True
    for text, audio in list:
        audio.seek(0)
        vrc_chatbox(text)
        play_sound(audio)
        audio.close()
    speaking = False


def tts(text):
    global speaking
    global panic
    speaking = True
    audioBytes = opts.tts_engine.tts(text)
    if audioBytes == None:
        speaking = False
        panic = True
        return
    play_sound(audioBytes)
    audioBytes.close()
    speaking = False


def to_wav(file, speed=1.0):
    """ Turns an .mp3 file into a .wav file (and optionally speeds it up) """
    name = file[0:file.rfind('.')]
    name = name + '.wav'
    try:
        start_time = time.perf_counter()
        input_stream = ffmpeg.input(file)
        audio = input_stream.audio.filter('atempo', speed)
        output_stream = audio.output(name, format='wav')
        ffmpeg.run(output_stream, cmd=[
                   "ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True, overwrite_output=True)
        end_time = time.perf_counter()
        verbose_print(f'--ffmpeg took {end_time - start_time:.3f}s')
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to convert audio: {e.stderr}") from e


def detect_silence(wf):
    """ Detects the duration of silence at the end of a wave file """
    threshold = 1024
    channels = wf.getnchannels()
    frame_rate = wf.getframerate()
    n_frames = wf.getnframes()
    if n_frames == 2147483647: 
        verbose_print("!!Something went bad trying to detect silence")
        return 0.0
    duration = n_frames / frame_rate

    # set the position to the end of the file
    wf.setpos(n_frames - 1)

    # read the last frame and convert it to integer values
    last_frame = wf.readframes(1)
    last_frame_values = struct.unpack("<h" * channels, last_frame)

    # check if the last frame is silent
    is_silent = all(abs(value) < threshold for value in last_frame_values)

    if is_silent:
        # if the last frame is silent, continue scanning backwards until a non-silent frame is found
        while True:
            # move the position backwards by one frame
            wf.setpos(wf.tell() - 2)

            # read the current frame and convert it to integer values
            current_frame = wf.readframes(1)
            current_frame_values = struct.unpack("<h" * channels, current_frame)

            # check if the current frame is silent
            is_silent = all(abs(value) < threshold for value in current_frame_values)

            if not is_silent:
                # if a non-silent frame is found, calculate the duration of the silence at the end
                silence_duration = duration - (wf.tell() / frame_rate)
                return silence_duration
            elif wf.tell() == 0:
                # if the beginning of the file is reached without finding a non-silent frame, assume the file is silent
                return duration
    else:
        # if the last frame is not silent, assume the file is not silent
        return 0.0


def clip_audio_end(audio_bytes: BytesIO) -> BytesIO:
    """Trims the end of audio in a BytesIO object"""
    audio_bytes.seek(0)
    with wave.open(audio_bytes, mode='rb') as wf:
        channels, sample_width, framerate, nframes = wf.getparams()[:4]
        duration = nframes / framerate
        silence_duration = detect_silence(wf)
        trimmed_length = int((duration - silence_duration + 0.050) * framerate)
        if trimmed_length <= 0:
            return BytesIO(b'RIFF\x00\x00\x00\x00WAVE')
        wf.setpos(0)
        output_bytes = BytesIO()
        with wave.open(output_bytes, mode='wb') as output_wf:
            output_wf.setnchannels(channels)
            output_wf.setsampwidth(sample_width)
            output_wf.setframerate(framerate)
            output_wf.writeframes(wf.readframes(trimmed_length))
        output_bytes.seek(0)
        return output_bytes


def handle_command(command):
    """ Handle voice commands """
    match command:
        case 'reset':
            opts.message_array = []
            print(f'$ Messages cleared!')
            vrc_chatbox('🗑️ Cleared message buffer')
            play_sound('./prebaked_tts/Clearedmessagebuffer.wav')

        case 'chatbox':
            opts.chatbox = not opts.chatbox
            ui.app.program_bools_frame.update_checkboxes()
            print(f'$ Chatbox set to {opts.chatbox}')
            play_sound(
                f'./prebaked_tts/Chatboxesarenow{"on" if opts.chatbox else "off"}.wav')

        case 'sound':
            opts.soundFeedback = not opts.soundFeedback
            ui.app.program_bools_frame.update_checkboxes()
            print(f'$ Sound feedback set to {opts.soundFeedback}')
            vrc_chatbox(('🔊' if opts.soundFeedback else '🔈') +
                        ' Sound feedback set to ' + ('on' if opts.soundFeedback else 'off'))
            play_sound(
                f'./prebaked_tts/Soundfeedbackisnow{"on" if opts.soundFeedback else "off"}.wav')

        case 'audiotrigger':
            opts.audio_trigger_enabled = not opts.audio_trigger_enabled
            ui.app.program_bools_frame.update_checkboxes()
            print(f'$ Audio Trigger set to {opts.audio_trigger_enabled}')
            vrc_chatbox(('🔊' if opts.audio_trigger_enabled else '🔈') +
                        ' Audio Trigger set to ' + ('on' if opts.audio_trigger_enabled else 'off'))
            # play_sound(f'./prebaked_tts/Audiotriggerisnow{"on" if audio_trigger_enabled else "off"}.wav')

        case 'messagelog':
            print(f'{opts.message_array}')
            vrc_chatbox('📜 Dumped messages, check console')
            play_sound('./prebaked_tts/DumpedmessagesCheckconsole.wav')

        case 'verbose':
            opts.verbosity = not opts.verbosity
            ui.app.program_bools_frame.update_checkboxes()
            print(f'$ Verbose logging set to {opts.verbosity}')
            vrc_chatbox('📜 Verbose logging set to ' +
                        ('on' if opts.verbosity else 'off'))
            play_sound(
                f'./prebaked_tts/Verboseloggingisnow{"on" if opts.verbosity else "off"}.wav')

        case 'shutdown':
            print('$ Shutting down...')
            vrc_chatbox('👋 Shutting down...')
            play_sound('./prebaked_tts/OkayGoodbye.wav')
            sys.exit(0)

        case 'gpt3':
            opts.gpt = 'GPT-3.5-Turbo'
            ui.app.ai_stuff_frame.update_radio_buttons()
            print(f'$ Now using {opts.gpt}')
            vrc_chatbox('Now using GPT-3.5-Turbo')
            play_sound('./prebaked_tts/NowusingGPT35Turbo.wav')

        case 'gpt4':
            opts.gpt = 'GPT-4'
            ui.app.ai_stuff_frame.update_radio_buttons()
            print(f'$ Now using {opts.gpt}')
            vrc_chatbox('Now using GPT-4')
            play_sound('./prebaked_tts/NowusingGPT4.wav')

        case 'parrotmode':
            opts.parrot_mode = not opts.parrot_mode
            ui.app.program_bools_frame.update_checkboxes()
            print(f'$ Parrot mode set to {opts.parrot_mode}')
            vrc_chatbox(
                f'🦜 Parrot mode is now {"on" if opts.parrot_mode else "off"}')
            play_sound(
                f'./prebaked_tts/Parrotmodeisnow{"on" if opts.parrot_mode else "off"}.wav')

        case 'thesenutsinyourmouth':
            vrc_chatbox('🤖 Do you like Imagine Dragons?')
            play_sound('./prebaked_tts/DoyoulikeImagineDragons.wav')
            time.sleep(3)
            vrc_chatbox('🤖 Imagine Dragon deez nuts across your face 😈')
            play_sound('./prebaked_tts/ImagineDragondeeznutsacrossyourface.wav')

        # If an exact match is not confirmed, this last case will be used if provided
        case _:
            print(f"$Unknown command: {command}")
            play_sound('./prebaked_tts/Unknowncommand.wav')


def default_handler(address, *args):
    """ Default handler for OSC messages received from VRChat """
    print(f"{address}: {args}")


def parameter_handler(address, *args):
    """ Handle OSC messages for specific parameters received from VRChat """
    global trigger
    if address == "/avatar/parameters/ChatGPT_PB" or address == "/avatar/parameters/ChatGPT":
        if args[0]:
            trigger = True
        verbose_print(f"{address}: {args} (V:{trigger})")


def vrc_chatbox(message):
    """ Send a message to the VRC chatbox if enabled """
    if opts.chatbox:
        vrc_osc_client.send_message("/chatbox/input", [message, True, False])


def vrc_set_parameter(address, value):
    """ Sets an avatar parameter on your current VRC avatar """
    address = "/avatar/parameters/" + address
    vrc_osc_client.send_message(address, value)


def check_doublepress_key(key):
    """ Check if ctrl key is pressed twice within a certain time window """
    global key_press_window_timeup
    global trigger
    global panic
    if key == opts.key_trigger_key:
        if speaking: panic = True
        if time.time() > key_press_window_timeup:
            key_press_window_timeup = time.time() + opts.key_press_window
        else:
            if (not recording) and (not speaking):
                trigger = True


# (thread target) Initialize Faster Whisper and move its model to the GPU if possible
def load_whisper():
    global model
    with whisper_lock:
        verbose_print("~Attempt to load Whisper...")
        vrc_chatbox('🔄 Loading Voice Recognition...')
        model = None
        start_time = time.perf_counter()
        model = WhisperModel(opts.whisper_model, device='cuda', compute_type="int8") # FasterWhisper
        end_time = time.perf_counter()
        verbose_print(f'--Whisper loaded in {end_time - start_time:.3f}s')
        vrc_chatbox('✔️ Voice Recognition Loaded')


def init_audio():
    global vb_in
    global vb_out
    global pyAudio
    pyAudio = pyaudio.PyAudio()
    info = pyAudio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    # vrc_chatbox('🔢 Enumerating Audio Devices...')
    # Get VB Aux Out for Input to Whisper, and VB Aux In for mic input
    start_time = time.perf_counter()
    for i in range(numdevices):
        info = pyAudio.get_device_info_by_host_api_device_index(0, i)
        if (info.get('maxInputChannels')) > 0:
            if info.get('name').startswith(opts.in_dev_name):
                verbose_print("~Found Input Device")
                verbose_print( info.get('name') )
                vb_out = i
        if (info.get('maxOutputChannels')) > 0: 
            if info.get('name').startswith(opts.out_dev_name):
                verbose_print("~Found Output Device")
                verbose_print( info.get('name') )
                vb_in = i
        if vb_in is not None and vb_out is not None: break
    if vb_out is None:
        print("!!Could not find input device for mic. Exiting...")
        raise RuntimeError
    if vb_in is None:
        print("!!Could not find output device for tts. Exiting...")
        raise RuntimeError

    end_time = time.perf_counter()
    verbose_print(f'--Audio initialized in {end_time - start_time:.5f}s')


# Program Setup #################################################################################################################################

# VRC OSC init
# Client (Sending)
vrc_osc_client = udp_client.SimpleUDPClient(ip, inPort)
vrc_chatbox('▶️ Starting...')
# Server (Receiving)
dispatcher = Dispatcher()
dispatcher.map("/avatar/parameters/*", parameter_handler)
vrc_osc_server = ThreadingOSCUDPServer((ip, outPort), dispatcher)


# Audio setup
init_audio()
# Create the stream to record user voice
streamIn = pyAudio.open(format=FORMAT,
                  channels=2,
                  rate=RATE,
                  input=True,
                  input_device_index=vb_out,
                  frames_per_buffer=CHUNK_SIZE)


# Main loop - Wait for sound. If sound heard, record frames to wav file,
#     then transcribe it with Whisper, then send that to ChatGPT, then
#     take the text from ChatGPT and play it through TTS
def loop():
    # TODO: fix this global bullshit
    global full_end_time
    global frames
    global lastFrame
    global recording
    global trigger
    global silence_timeout_timer
    global panic

    opts.LOOP = True

    full_end_time = time.perf_counter()
    print(f'--Program init took {full_end_time - full_start_time:.3f}s')

    while model is None:
        time.sleep(0.1)
        pass

    print("~Waiting for sound...")
    while opts.LOOP:
        try:
            data = streamIn.read(CHUNK_SIZE)
            # calculate root mean square of audio data
            rms = audioop.rms(data, 2)

            if opts.audio_trigger_enabled:
                if (not recording and rms > opts.THRESHOLD):
                    trigger = True

            # Start recording if sound goes above threshold or parameter is triggered
            if not recording and trigger:
                if lastFrame is not None:
                    # Add last frame to buffer, in case the next frame starts recording in the middle of a word
                    frames.append(lastFrame)
                frames.append(data)
                vrc_chatbox('👂 Listening...')
                verbose_print("~Recording...")
                recording = True
                # set timeout to now + SILENCE_TIMEOUT seconds
                silence_timeout_timer = time.time() + opts.SILENCE_TIMEOUT
                if opts.soundFeedback:
                    play_sound_threaded(speech_on)
            elif recording:  # If already recording, continue appending frames
                frames.append(data)
                if rms < opts.THRESHOLD:
                    if time.time() > silence_timeout_timer:  # if silent for longer than SILENCE_TIMEOUT, save
                        verbose_print("~Saving (silence)...")
                        recording = False
                        trigger = False
                        save_recorded_frames(frames)
                        verbose_print("~Waiting for sound...")
                        frames = []
                        panic = False
                else:
                    # set timeout to now + SILENCE_TIMEOUT seconds
                    silence_timeout_timer = time.time() + opts.SILENCE_TIMEOUT

                # if recording for longer than MAX_RECORDING_TIME, save
                if len(frames) * CHUNK_SIZE >= opts.MAX_RECORDING_TIME * RATE:
                    verbose_print("~Saving (length)...")
                    recording = False
                    trigger = False
                    save_recorded_frames(frames)
                    verbose_print("~Waiting for sound...")
                    frames = []
                    panic = False

            lastFrame = data
            # time.sleep(0.001)  # sleep to avoid burning cpu
        except Exception as e:
            print(f'!!Exception:\n{e}')
            vrc_chatbox(f'⚠ {e}')
            streamIn.close()
            opts.LOOP = False
            sys.exit(e)
        except KeyboardInterrupt:
            print('Keyboard interrupt')
            vrc_chatbox(f'⚠ Quitting')
            streamIn.close()
            vrc_osc_server.shutdown()
            opts.LOOP = False
            sys.exit("KeyboardInterrupt")
    print("Exiting, Bye!")
    streamIn.close()
    vrc_osc_server.shutdown()

def start_server(server):  # (thread target) Starts OSC Listening server
    verbose_print(f'~Starting OSC Listener on {ip}:{outPort}')
    server.serve_forever()


def start_key_listener():  # (thread target) Starts Keyboard Listener
    with Listener(on_release=check_doublepress_key) as listener:
        listener.join()


def start_ui(): # (thread target) Starts GUI
    ui.initialize()

whisper_thread = threading.Thread(name='whisper-thread', target=load_whisper)
serverThread = threading.Thread(
    name='oscserver-thread', target=start_server, args=(vrc_osc_server,), daemon=True)
key_listener_thread = threading.Thread(name='keylistener-thread', target=start_key_listener, daemon=True)
uithread = threading.Thread(name="ui-thread", target=start_ui, daemon=True)
mainLoopThread = threading.Thread(name='mainloop-thread', target=loop)

whisper_thread.start()
serverThread.start()
key_listener_thread.start()
whisper_thread.join()  # Wait for Whisper to be loaded first before trying to use it
uithread.start()
mainLoopThread.start()
uithread.join()
sys.exit(0)
