import asyncio
import concurrent.futures
import io
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import Audio, ChatCompletion
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Button, Static, TextLog, Label

# 2. Write logic for chat stop phrase "What do you think?"
# 3. Write logic for command stop phrase "Make it so."
# 4. Add checkbox for real-time transcription
# 5. Add checkbox for audio playback

DEVICE_NAME = "MacBook Pro Microphone"
STT_MODEL_NAME = "whisper-1"
LLM_MODEL_NAME = "gpt-3.5-turbo"
MIN_DURATION = 2.0
THRESHOLD = 0.01
SYSTEM_PROMPT = """
You are Jarvis, an interviewer who listens and participates in dialogues to
help people develop their creative ideas. Your goal is to create an unusually
interesting conversation with lots of specific details. Do not speak in
generalities or cliches. I'd like you to have a dialogue with me about an idea
that I have.
"""

class Transcription:
    """A class for storing and manipulating transcriptions."""
    def __init__(self):
        self.transcript_chunks = []
        self.transcript_full = ""

def transcribe(audio: np.ndarray, sample_rate: int, prompt: str = "") -> str:
    """Transcribe audio data and return transcript"""
    with io.BytesIO() as memory_file:
        memory_file.name = "stream.wav"
        sf.write(memory_file, audio, sample_rate, format="WAV")
        memory_file.seek(0)
        response = Audio.transcribe(STT_MODEL_NAME, 
                                    memory_file, 
                                    prompt=prompt)
        return response.text

async def transcribe_audio_chunk(transcription: Transcription,
                                 audio: np.ndarray,
                                 sample_rate: int) -> None:
    prompt = str.join(" ", transcription.transcript_chunks)
    transcription.transcript_chunks.append("...")
    task_id = len(transcription.transcript_chunks) - 1

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        transcript = await loop.run_in_executor(pool, 
                                                transcribe,
                                                audio,
                                                sample_rate,
                                                prompt)
        print(f"audio chunk: {transcript}")
        transcription.transcript_chunks[task_id] = transcript
    
async def transcribe_audio_full(transcription: Transcription,
                                audio: np.ndarray,
                                sample_rate: int) -> None:
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        transcript = await loop.run_in_executor(pool, 
                                                transcribe,
                                                audio,
                                                sample_rate)
        print(f"audio full: {transcript}")
        transcription.transcript_full = transcript

async def record(transcription: Transcription, 
                 device_name: str,
                 sample_rate: int,
                 stop_recording: asyncio.Event) -> None:
    """Record audio and transcribe audio in the background using device"""
    audio_chunk = None
    audio_full = None
    q = queue.Queue()

    def callback(indata, frames, time, status):
        nonlocal audio_chunk, audio_full
        audio_data = np.frombuffer(indata, dtype="float32")
        if audio_chunk is None:
            audio_chunk = audio_data
        else:
            rms = np.sqrt(np.mean(np.square(audio_data)))
            if rms < THRESHOLD:
                duration = audio_chunk.size / sample_rate
                if duration > MIN_DURATION:
                    print(f"asyncing an audio chunk")
                    q.put(audio_chunk.copy())
                    if audio_full is None:
                        audio_full = audio_chunk.copy()
                    else:
                        audio_full = np.append(audio_full, audio_chunk)
                    audio_chunk = None
            else:
                audio_chunk = np.append(audio_chunk, audio_data)

    with sd.InputStream(samplerate=sample_rate, 
                        blocksize=int(sample_rate / 2),
                        device=device_name,
                        callback=callback):
        while not stop_recording.is_set():
            while not q.empty():
                audio = q.get()
                await transcribe_audio_chunk(transcription, 
                                             audio, 
                                             sample_rate)
            await asyncio.sleep(0.1)
        if audio_chunk is not None:
            await transcribe_audio_chunk(transcription, 
                                         audio_chunk, 
                                         sample_rate)
        if audio_full is None:
            audio_full = audio_chunk
        await transcribe_audio_full(transcription, audio_full, sample_rate)

async def write_transcript(transcription: Transcription,
                           output: Static) -> None:
    """Write transcription chunks incrementally to output widget"""
    while True:
        if len(transcription.transcript_chunks) > 0:
            full_text = str.join(" ", transcription.transcript_chunks)
            output.update(full_text)
        await asyncio.sleep(0)

class App(App):
    CSS_PATH = "style.css"
    TITLE = "Jarvis"
    SUB_TITLE = "Speak to Jarvis"
    BINDINGS = [
        ("q", "quit", "Quit the application")
    ]

    async def on_mount(self) -> None:
        self.device = sd.query_devices(DEVICE_NAME)
        self.stop_recording_event = asyncio.Event()
        self.recording = False
        self.transcription = Transcription()
        self.history = []

        transcript = self.query_one("#transcript")
        self.write_task = asyncio.create_task(
            write_transcript(self.transcription,
                             transcript)
        )

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("You:"),
            Static("...", id="transcript", classes="input"),
            Button("Record", id="start_stop"),
            TextLog(id="log", wrap=True)
        )
        yield Footer()

    def action_quit(self) -> None:
        self.exit()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start_stop":
            if not self.recording:
                self.recording = True
                self.transcription.transcript_chunks.clear()
                self.transcription.transcript_full = ""
                self.stop_recording_event.clear()
                event.button.label = "Stop"
                device_name = self.device["name"]
                sample_rate = int(self.device["default_samplerate"])
                self.record_task = asyncio.create_task(
                    record(self.transcription, 
                          device_name, 
                          sample_rate,
                          self.stop_recording_event)
                )
            else:
                self.recording = False
                self.stop_recording_event.set()
                event.button.label = "Record"

                await self.record_task
                full_text = self.transcription.transcript_full

                text_log = self.query_one("#log")
                text_log.write(f"YOU: {full_text}")

                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                if len(self.history) > 0:
                    messages += self.history

                messages.append({"role": "user", "content": full_text})
                response = ChatCompletion.create(
                    model=LLM_MODEL_NAME,
                    messages = messages
                )
                answer = response.choices[0].message.content
                text_log.write(f"JARVIS: {answer}")
                self.history.append({"role": "user", "content": full_text})
                self.history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    app = App()
    app.run()