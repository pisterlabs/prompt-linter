import torch
import pyaudio
import webrtcvad
import numpy as np
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from elevenlabs import generate
import threading
import multiprocessing as mp
import subprocess
from typing import Iterator
import torchaudio
import itertools
import random


ELEVEN_API_KEY = 'not a chance'
STT_MODEL = None
PYAUDIO = None
VAD = None
ANTH_CLIENT = Anthropic(api_key="I don't think so")
CONVERSATION = ''
TTS_QUEUE = mp.Queue()  # Create a queue object


def stream(audio_stream: Iterator[bytes]):
    #mpv_command = ["mpv", "--demuxer=rawaudio", "--demuxer-rawaudio-format=s16le", "--demuxer-rawaudio-rate=48000", "--demuxer-rawaudio-channels=1", "--no-cache", "--no-terminal", "--", "fd://0"]
    mpv_command = ['mpv', '--no-cache', '--no-terminal', '--', 'fd://0']
    mpv_process = subprocess.Popen(
        mpv_command,
        stdin=subprocess.PIPE,
        #stdout=subprocess.DEVNULL,
        #stderr=subprocess.DEVNULL,
    )

    audio = b""

    for chunk in audio_stream:
        if chunk is not None:
            mpv_process.stdin.write(chunk)  # type: ignore
            mpv_process.stdin.flush()  # type: ignore
            audio += chunk

    if mpv_process.stdin:
        mpv_process.stdin.close()
    
    mpv_process.wait()


def runTTSElevenLabsStreamed(TTS_QUEUE, ELEVEN_API_KEY): #TODO: elevenlabs chunking is dumb and I need to either do my own streaming or find a better tts
    def timed_iterator(original_iterator, initTime):
        for idx, chunk in enumerate(original_iterator):
            if idx == 0:  # First chunk
                print(f"TTS took {time.time() - initTime} seconds.")
            yield chunk

    def run_queue(starter, audioStart):
        yield starter
        print('yielded starter in' , time.time()-audioStart)
        item = TTS_QUEUE.get()
        while item is not None:
            yield item
            print('yielded item in' , time.time()-audioStart)
            item = TTS_QUEUE.get()

    while True:
        buffer = TTS_QUEUE.get()
        print('starting tts with', buffer)
        audioStart = time.time()

        audio_stream = generate(
            api_key=ELEVEN_API_KEY,
            text=run_queue(buffer, audioStart),
            voice="Daniel",
            stream=True,
            latency=4,
        )

        stream(timed_iterator(audio_stream, audioStart))
        print('ended TTS')


def runTTSElevenLabs(TTS_QUEUE, ELEVEN_API_KEY):
    def chunkToWords(iterator):
        start = time.time()
        buffer = ''
        for chunk in iterator:
            buffer += chunk
            if buffer.endswith('.') or buffer.endswith('?') or buffer.endswith('!') or buffer.endswith(' '):
                if start:
                    print('chunking took', time.time()-start)
                    start = None
                yield buffer
                buffer = ''

    def streamTTS(iterator):
        start = time.time()
        for chunk in iterator:
            audio = generate(
                api_key=ELEVEN_API_KEY,
                text=chunk,
                voice="Daniel",
                stream=True,
                latency=4,
            )
            if start:
                print('TTS took', time.time()-start)
                start = None
            print(audio)
            yield audio

    while True:
        queueIter = iter(TTS_QUEUE.get, None)
        stream(itertools.chain.from_iterable(streamTTS(chunkToWords(queueIter))))
        print('ended TTS')


def runTTS(TTS_QUEUE, _):
    language = 'en'
    model_id = 'v3_en'
    sample_rate = 48000
    speaker = 'en_0'
    device = torch.device('cpu')

    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                        model='silero_tts',
                                        language=language,
                                        speaker=model_id)
    model.to(device)  # gpu or cpu

    def chunkToWords(iterator):
        start = time.time()
        buffer = ''
        for chunk in iterator:
            buffer += chunk
            if buffer.endswith('.') or buffer.endswith('?') or buffer.endswith('!') or buffer.endswith(' '):
                if start:
                    print('chunking took', time.time()-start)
                    start = None
                yield buffer
                buffer = ''

    def streamTTS(iterator):
        start = time.time()
        for chunk in iterator:
            audio = model.apply_tts(text=chunk,
                    speaker=speaker,
                    sample_rate=sample_rate)
            
            audio = audio.detach().cpu()
            int_samples = (audio * (2**15 - 1)).clamp(min=-2**15, max=2**15 - 1).short()
            audio_bytes = int_samples.numpy().tobytes()

            if start:
                print('TTS took', time.time()-start)
                start = None
            yield audio_bytes

    while True:
        queueIter = iter(TTS_QUEUE.get, None)
        stream(streamTTS(chunkToWords(queueIter)))
        print('ended TTS')


def callClaude(input):
    global CONVERSATION
    if not CONVERSATION:
        input = 'Please respond using at most five words. ' + input
    CONVERSATION += f"{HUMAN_PROMPT} {input}{AI_PROMPT}"

    start = time.time()
    stream = ANTH_CLIENT.completions.create(
        prompt=CONVERSATION,
        max_tokens_to_sample=100,
        model="claude-instant-1",
        stream=True,
    )

    def textStream(start):
        global CONVERSATION
        for completion in stream:
            if start:
                print("Claude took ", time.time()-start)
                start = None
                print(CONVERSATION)

            print(completion.completion, end="", flush=True)
            CONVERSATION += completion.completion
            TTS_QUEUE.put(completion.completion)
        
        print()

    while not TTS_QUEUE.empty():
        TTS_QUEUE.get()    
    textStream(start)
    TTS_QUEUE.put(None)
    listen()


def setupSTT():
    # Initialize PyAudio and WebRTC VAD
    global PYAUDIO, VAD
    PYAUDIO = pyaudio.PyAudio()
    VAD = webrtcvad.Vad(3)

    device = torch.device(
        "cpu"
    )  # gpu also works, but our models are fast enough for CPU
    return torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_stt",
        jit_model="jit_q",
        language="en",
        device=device,
    )


def startStockPhrases():
    print("Starting stock phrases...")
    stockPhrases = ['let me think.... ', 'interesting!', 'hmmmm', "thanks!", 'wow!', 'fun idea....', 'I like that!', 'I see...', 'I understand...',]

    
    TTS_QUEUE.put(random.choice(stockPhrases))


def getSpeech():
    RATE = 16000 #MUST be 16k for the model
    CHUNK_SIZE = 480   # 30ms
    SILENT_CHUNKS_THRESHOLD = 15
    MINIMUM_SPEECH_CHUNKS = 8

    audio_chunks = np.array([])
    num_silent_chunks = 0
    started_speaking = 0
    
    # Open microphone stream
    stream = PYAUDIO.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    while True:
        audio_chunk = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)
        is_speech = VAD.is_speech(audio_chunk.tobytes(), sample_rate=RATE) #TODO: use model as vad
        audio_chunks = np.concatenate((audio_chunks, audio_chunk))
        
        if is_speech:
            if not started_speaking % 5:
                print("Speaking...")
            started_speaking += 1
            num_silent_chunks = 0

        else:
            num_silent_chunks += 1
            if num_silent_chunks >= SILENT_CHUNKS_THRESHOLD and started_speaking > MINIMUM_SPEECH_CHUNKS:
                print("Finished speaking.")
                startStockPhrases()
                start = time.time()
                break

    # Stop the stream and close PyAudio
    stream.stop_stream()
    stream.close()

    waveform = torch.from_numpy(audio_chunks.astype(np.float32)).unsqueeze(0)
    waveform = waveform.to(torch.device("cpu"))

    print("Processing audio took: ", time.time()-start)
    start = time.time()
    output = STT_MODEL(waveform)
    return output, start


def listen():
    print("Listening...")

    speech = ''
    while not speech:
        output, start = getSpeech()

        if len(output) != 1:
            raise ValueError(
                f"Expected model to return 1 output, instead got {len(output)}"
            )  
        speech = STT_DECODER(output[0].cpu())
        print("STT took ", time.time()-start)

    print("You said: ", speech)

    callClaude(speech)

if __name__ == "__main__":
    audio_thread = mp.Process(target=runTTSElevenLabs, args=(TTS_QUEUE, ELEVEN_API_KEY))
    audio_thread.start()
    STT_MODEL, STT_DECODER, _ = setupSTT()
    listen()
    audio_thread.join()

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech


def transcribe_streaming_v2(
    project_id: str,
    audio_file: str,
) -> cloud_speech.StreamingRecognizeResponse:
    """Transcribes audio from audio file stream.

    Args:
        project_id: The GCP project ID.
        audio_file: The path to the audio file to transcribe.

    Returns:
        The response from the transcribe method.
    """
    # Instantiates a client
    client = SpeechClient()

    # Reads a file as bytes
    with open(audio_file, "rb") as f:
        content = f.read()

    # In practice, stream should be a generator yielding chunks of audio data
    chunk_length = len(content) // 5
    stream = [
        content[start : start + chunk_length]
        for start in range(0, len(content), chunk_length)
    ]
    audio_requests = (
        cloud_speech.StreamingRecognizeRequest(audio=audio) for audio in stream
    )

    recognition_config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="long",
    )
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=recognition_config
    )
    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=f"projects/{project_id}/locations/global/recognizers/_",
        streaming_config=streaming_config,
    )

    def requests(config: cloud_speech.RecognitionConfig, audio: list) -> list:
        yield config
        yield from audio

    # Transcribes the audio into text
    responses_iterator = client.streaming_recognize(
        requests=requests(config_request, audio_requests)
    )
    responses = []
    for response in responses_iterator:
        responses.append(response)
        for result in response.results:
            print(f"Transcript: {result.alternatives[0].transcript}")

    return responses
