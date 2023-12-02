import asyncio
import sys
import threading

from faster_whisper import WhisperModel
from audio_transcriber import AppOptions
from audio_transcriber import AudioTranscriber
from utils.audio_utils import get_valid_input_devices, base64_to_audio
from openai_api import OpenAIAPI
import time

# Code adapted from:
# https://github.com/reriiasu/speech-to-text/blob/main/speech_to_text/__main__.py


transcriber: AudioTranscriber = None
event_loop: asyncio.AbstractEventLoop = None
thread: threading.Thread = None
openai_api: OpenAIAPI = None


def start_transcription():
    global transcriber, event_loop, thread, openai_api
    try:
        filtered_model_settings = {'model_size_or_path': 'base.en', 'device': 'auto', 'device_index': 0, 'compute_type': 'default', 'cpu_threads': 0, 'num_workers': 1, 'local_files_only': False}
        filtered_app_settings = {'audio_device': 0, 'silence_limit': 8, 'noise_threshold': 5, 'non_speech_threshold': 0.1, 'include_non_speech': False, 'use_openai_api': False}
        filtered_transcribe_settings = {'language': 'en', 'task': 'transcribe', 'beam_size': 5, 'best_of': 5, 'patience': 1, 'length_penalty': 1, 'repetition_penalty': 1, 'no_repeat_ngram_size': 0, 'temperature': [0, 0.2, 0.4, 0.6, 0.8, 1], 'compression_ratio_threshold': 2.4, 'log_prob_threshold': -1, 'no_speech_threshold': 0.6, 'condition_on_previous_text': True, 'suppress_blank': True, 'suppress_tokens': [-1], 'without_timestamps': False, 'max_initial_timestamp': 1, 'word_timestamps': False, 'prepend_punctuations': '\\"\'“¿([{-', 'append_punctuations': '\\"\'.。,!?:：”)]}、', 'vad_filter': True, 'vad_parameters': {'threshold': 0.5, 'min_speech_duration_ms': 250, 'max_speech_duration_s': 0, 'min_silence_duration_ms': 2000, 'window_size_samples': 1024, 'speech_pad_ms': 400}}

        whisper_model = WhisperModel(**filtered_model_settings)
        app_settings = AppOptions(**filtered_app_settings)
        event_loop = asyncio.new_event_loop()

        if app_settings.use_openai_api:
            openai_api = OpenAIAPI()

        transcriber = AudioTranscriber(
            event_loop,
            whisper_model,
            filtered_transcribe_settings,
            app_settings,
            openai_api,
        )
        asyncio.set_event_loop(event_loop)
        thread = threading.Thread(target=event_loop.run_forever, daemon=True)
        thread.start()

        asyncio.run_coroutine_threadsafe(transcriber.start_transcription(), event_loop)
    except Exception as e:
        print("Error:", e)


def stop_transcription():
    global transcriber, event_loop, thread, openai_api
    if transcriber is None:
        return
    transcriber_future = asyncio.run_coroutine_threadsafe(
        transcriber.stop_transcription(), event_loop
    )
    transcriber_future.result()

    if thread.is_alive():
        event_loop.call_soon_threadsafe(event_loop.stop)
        thread.join()
    event_loop.close()
    transcriber = None
    event_loop = None
    thread = None
    openai_api = None


if __name__ == "__main__":
    start_transcription()
    time.sleep(50)
    if transcriber and transcriber.transcribing:
        stop_transcription()

        
