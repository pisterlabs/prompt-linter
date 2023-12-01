import whisper
import openai
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from threading import Thread
from pydub import AudioSegment
import speech_recognition as sr
from pprint import pprint
import asyncio
from tqdm import tqdm
from secret import OPENAI_API_KEY
import os
import sys
sys.path.append('../')


class PunctuationPostprocessing:
    def __init__(self):
        self.model_name = "junsun10/kobart-base-v2-add-period"
        self.model = BartForConditionalGeneration.from_pretrained(
            self.model_name).to("cuda")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.model_name)

    def punctuation_inference(self, text):
        split_text = [text[i:i+100] for i in range(0, len(text), 100)]
        result = []
        print("[KoBART Punctuation PostProcessing]")
        for text in tqdm(split_text):
            input_ids = self.tokenizer.encode(
                text, return_tensors="pt").to("cuda")
            punctuated_tokens = self.model.generate(
                input_ids=input_ids,
                bos_token_id=self.model.config.bos_token_id,
                eos_token_id=self.model.config.eos_token_id,
                length_penalty=2.0,
                max_length=50,
                num_beams=4,
            )
            punctuated_text = self.tokenizer.decode(
                punctuated_tokens[0], skip_special_tokens=True)
            result.append([text, punctuated_text])
        return result

    def remove_wrong_results(self, result):
        new_result = ""
        for origin, predict in result:
            origin_count = origin.count(".")
            predict_count = predict.count(".")
            new_predict = predict[:len(
                origin) + predict_count - origin_count - 1]
            new_result += new_predict
        return new_result

    def postprocess(self, text):
        result = self.punctuation_inference(text)
        new_result = self.remove_wrong_results(result)
        return new_result


def split_audio_by_time(audio_segment, chunk_duration=3):
    # Convert chunk duration from minutes to milliseconds
    chunk_length = chunk_duration * 60 * 1000
    chunks = []
    for i in range(0, len(audio_segment), chunk_length):
        chunk = audio_segment[i:i+chunk_length]
        chunks.append(chunk)
    return chunks


def milliseconds_to_time(milliseconds):
    seconds = milliseconds / 1000
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    formatted_time = f"{hours}h {minutes}m {seconds:.2f}s" if hours > 0 else f"{minutes}m {seconds:.2f}s"
    return formatted_time


def whisper_transcribe(audio_path, whisper_version, wav_path=None):
    audio_format = audio_path.split('.')[-1]
    audio_segment = AudioSegment.from_file(audio_path, format=audio_format)

    if wav_path:
        wav_dir = '/'.join(wav_path.split('/')[:-1])
        os.makedirs(wav_dir, exist_ok=True)
        audio_segment.export(wav_path, format='wav')

    print(f"[Whisper Transcription ({whisper_version})]")
    print(
        f"Audio Length (Time): {milliseconds_to_time(len(audio_segment))} ({len(audio_segment)} ms)")
    print(f"Audio Frame Rate: {audio_segment.frame_rate / 1000} kHz")
    print(f"Audio Sample Width: {2**(audio_segment.sample_width+1)}-bit")

    assert "HOSTED" in whisper_version or whisper_version == "API", \
        "Please write appropriate Whisper's version (HOSTED_LARGE/MEDIUM/SMALL or API)"

    if "HOSTED" in whisper_version:
        stt_result = whisper_transcribe_hosted(audio_path)

    elif whisper_version == "API":
        chunk_duration = 2
        wav_chunks = split_audio_by_time(audio_segment, chunk_duration)

        stt_parts, threads = [], []
        for idx, chunk in enumerate(wav_chunks):
            print(f"Processing chunk {idx+1}...")
            thread = Thread(target=multi_worker, args=(chunk, idx, stt_parts))
            thread.start()
            threads.append(thread)

        for thread in tqdm(threads):
            thread.join()

        stt_parts = [stt_part for _, stt_part in sorted(stt_parts)]
        stt_result = " ".join(stt_parts)

    return stt_result


def whisper_transcribe_hosted(audio_path, whisper_version='LARGE'):
    version_mapping = {
        "LARGE": "large-v2",
        "MEDIUM": "medium",
        "SMALL": "small"
    }
    version = version_mapping[whisper_version.split('_')[-1]]
    model = whisper.load_model(version)
    stt_result = model.transcribe(audio_path)["text"]

    return stt_result


def whisper_transcribe_api(audio_segment):
    openai.api_key = OPENAI_API_KEY

    recognizer = sr.Recognizer()
    audio_data = sr.AudioData(
        audio_segment.raw_data, audio_segment.frame_rate, audio_segment.sample_width)
    stt_result = recognizer.recognize_whisper_api(
        audio_data, api_key=OPENAI_API_KEY)

    return stt_result


def multi_worker(chunk, idx, results_list):
    result = whisper_transcribe_api(chunk)
    if result:
        results_list.append((idx, result))


def transcribe(audio_path):
    """
    STT Models (whisper_version)
    - HOSTED_SMALL: OpenAI Whisper hosted small (price=free, no_limit)
    - HOSTED_MEDIUM: OpenAI Whisper hosted medium (price=free, no_limit)
    - HOSTED_LARGE: OpenAI Whisper hosted large-v2 (price=free, no_limit)
    - API: OpenAI Whisper API (price=0.006$/min, limit=25MB, same model as HOSTED LARGE)
    """
    whisper_version = "API"
    transcription = whisper_transcribe(
        audio_path, whisper_version, wav_path=None)
    postprocessor = PunctuationPostprocessing()
    transcription = postprocessor.postprocess(transcription)
    return transcription


async def transcribe_async(audio_path):
    """
    STT Models (whisper_version)
    - HOSTED_SMALL: OpenAI Whisper hosted small (price=free, no_limit)
    - HOSTED_MEDIUM: OpenAI Whisper hosted medium (price=free, no_limit)
    - HOSTED_LARGE: OpenAI Whisper hosted large-v2 (price=free, no_limit)
    - API: OpenAI Whisper API (price=0.006$/min, limit=25MB, same model as HOSTED LARGE)
    """
    whisper_version = "API"
    transcription = whisper_transcribe(
        audio_path, whisper_version, wav_path=None)
    postprocessor = PunctuationPostprocessing()
    transcription = postprocessor.postprocess(transcription)
    return transcription


async def transcribe_test(audio_path):
    await asyncio.sleep(5)
    transcription = "Sample Transcript"
    return transcription


def main():
    audio_path = "/opt/ml/final/stt/data/wav_test/sample/test_1m.m4a"
    transcription = transcribe(audio_path)
    pprint(transcription)


if __name__ == "__main__":
    main()
