import sys
import select
import ffmpeg
import pyaudio
import openai
from whispercpp import Whisper

# Define audio buffering parameters

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


def buffer_audio() -> bytes:
    """
    Buffers audio from the default microphone until the user presses enter
    :return: A bytes object containing the raw PCM data
    """

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* buffering audio")
    print("* press ENTER to stop buffering")

    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        # Check if stdin has recieved any input (indicating a newline was inputted, which would flush the stdin buffer)
        # This only works on *nix systems
        # This is used instead of the keyboard library because it doesn't require root
        if select.select([sys.stdin, ], [], [], 0.0)[0]:
            break

    print("* done buffering")

    stream.stop_stream()
    stream.close()
    p.terminate()

    raw_pcm = b''.join(frames)

    print("Converting...")
    try:
        process = (
            ffmpeg.input("pipe:", format="s16le", ac=CHANNELS, ar=RATE, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run_async(
                cmd=["ffmpeg"],
                pipe_stdin=True, quiet=True, pipe_stdout=True, pipe_stderr=True
            )
        )
        y, _ = process.communicate(input=raw_pcm)
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    
    return y

def convert_audio_to_pcm(filename: str) -> bytes:
    """
    Converts an audio file to raw PCM data

    :param filename: The filename of the audio file to convert
    :return: A bytes object containing the raw PCM data
    """
    print("Converting...")
    y, _ = (
            ffmpeg.input(filename, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(
                cmd=["ffmpeg"],
                quiet=True
            )
        )
    return y

def transcribe_audio(pcm: bytes, model: str) -> str:
    """
    Transcribes a PCM audio file with OpenAI Whisper
    
    :param pcm: A bytes object containing the raw PCM data
    :param model: The model to use for transcription. Valid options are:
                  - 'tiny.en': Tiny English model
                  - 'tiny': Tiny multilingual model
                  - 'small.en': Small English model
                  - 'small': Small multilingual model
                  - 'base.en': Base English model
                  - 'medium': Medium multilingual model
                  - 'large': Large multilingual model
    
    :type pcm: bytes
    :type model: str
    :return: The transcription result as a string
    """

    w = Whisper.from_pretrained(model)

    transcription = w.transcribe(pcm)
    return transcription

def generate_notes(transcription: str) -> str:
    """
    Generates lecture notes from a transcription. This function expects an API key to be defined in the OPENAI_API_KEY environment variable.

    :param transcription: The transcription to generate notes from
    :return: The generated notes
    """

    system_message = """You are an assistant designed to generate college student notes from lecture transcripts. These transcripts may contain errors, do your best to understand them anyways. Your notes MUST be in Markdown format, and should be useful material for students to review after class. These notes should be clearly broken down into sections, and should include enough information that a student who left the room to use the bathroom can easily catch up on what they missed while they were gone. The more information included, the better."""

    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[
        {"role": "system", "content": system_message}, {"role": "user", "content": transcription}])
    
    return chat_completion.choices[0].message.content
