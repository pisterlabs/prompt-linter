import sounddevice as sd
import numpy as np
import wave
import threading
import openai
client = openai.OpenAI()

VOLUME_THRESHOLD = 1.0
MIN_SILENCE_DURATION = 3.0  # in seconds
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MIN_SOUND_DURATION = 0.5  # in seconds


def detect_silence_and_save(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, output_file='output.wav'):
    """Detects silence from the microphone and saves the recording to a WAV file.

    Args:
        sample_rate (int): The sample rate to use for the microphone.
        chunk_size (int): The size of chunks to process at a time.
        output_file (str): The path to the WAV file to save the recording.
    """
    with sd.InputStream(samplerate=sample_rate, channels=1) as stream, wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for 'int16' type
        wf.setframerate(sample_rate)
        

        sound_duration = 0
        silence_duration = 0
        print('Please start speaking now...')
        recorded_frames = []
        state = "waiting"
        while True:
            data, _ = stream.read(chunk_size)
            volume = np.linalg.norm(data) * 10
            if state == "waiting":
                if volume > VOLUME_THRESHOLD:
                    print("Started speaking")
                    state = "recording"
                    recorded_frames.append(data)
                    sound_duration += chunk_size / sample_rate
                    continue
            if state == "recording":
                # print(volume, VOLUME_THRESHOLD)
                if volume < VOLUME_THRESHOLD:
                    silence_duration += chunk_size / sample_rate
                else:
                    sound_duration += chunk_size / sample_rate
                    silence_duration = 0
                recorded_frames.append(data)
                if silence_duration >= MIN_SILENCE_DURATION:
                    if sound_duration > MIN_SOUND_DURATION:
                        print('Silence detected, transcribing...')
                        break
                    else:
                        silence_duration = 0
                        sound_duration = 0
                        recorded_frames = []
                        state = "waiting"
                        continue
        audio_data = np.concatenate(recorded_frames, axis=0)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

def listen_until_enter_and_save(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, output_file='output.wav'):
    stop_recording = threading.Event()

    def record():
        with sd.InputStream(samplerate=sample_rate, channels=1) as stream, wave.open(output_file, 'wb') as wf:
            devicename = sd.query_devices(stream.device)['name']
            print(f'Please start speaking now on [{devicename}]... (hit ENTER to send)')
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for 'int16' type
            wf.setframerate(sample_rate)

            recorded_frames = []
            while not stop_recording.is_set():
                data, _ = stream.read(chunk_size)
                volume = np.linalg.norm(data) * 10
                print(f"volume: {int(volume)}", end="\r")
                recorded_frames.append(data)

            audio_data = np.concatenate(recorded_frames, axis=0)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    record_thread = threading.Thread(target=record)
    record_thread.start()

    input()
    stop_recording.set()
    record_thread.join()

def listen_and_transcribe(detectsilence=False):
    filename = '/tmp/gptexecsound.wav'
    if detectsilence:
        detect_silence_and_save(output_file=filename)
    else:
        listen_until_enter_and_save(output_file=filename)
    with open(filename, 'rb') as audio_data:
        transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_data)
    return transcription.text
