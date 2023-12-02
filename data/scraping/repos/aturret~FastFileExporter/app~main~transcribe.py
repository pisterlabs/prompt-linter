import os

from flask import request, jsonify, current_app

from app.main import main

from pydub import AudioSegment
import openai

TRANSCRIBE_MODEL = "whisper-1"
SEGMENT_LENGTH = 5 * 60


@main.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        data = request.get_json()
        audio_file = data.get("audio_file")
        openai_api_key = data.get("openai_api_key")
        transcript = get_audio_text(audio_file, openai_api_key)
        return jsonify({"transcript": transcript,
                        "message": "ok"}), 200
    except Exception as e:
        return jsonify({"message": f"error{str(e)}"}), 500


def get_audio_text(audio_file: str, openai_api_key: str):
    openai.api_key = openai_api_key
    transcript = ""
    AudioSegment.converter = "ffmpeg"
    audio_item = AudioSegment.from_file(audio_file, "m4a")
    start_trim = milliseconds_until_sound(audio_item)
    audio_item = audio_item[start_trim:]
    audio_length = int(audio_item.duration_seconds) + 1
    for index, i in enumerate(range(0, SEGMENT_LENGTH * 1000, audio_length * 1000)):
        start_time = i
        end_time = (i + SEGMENT_LENGTH) * 1000
        if end_time < audio_length * 1000:
            audio_segment = audio_item[start_time:]
        else:
            audio_segment = audio_item[start_time:end_time]
        audio_file_list = audio_file.split(".")
        audio_file_ext = audio_file_list[-1]
        audio_file_non_ext = ".".join(audio_file_list[:-1])
        audio_segment_path = audio_file_non_ext + "-" + str(index + 1) + "." + audio_file_ext
        audio_segment.export(audio_segment_path)
        print(f"audio_segment_path: {audio_segment_path}")
        audio_segment_file = open(audio_segment_path, "rb")
        transcript_segment = openai.Audio.transcribe(TRANSCRIBE_MODEL, audio_segment_file)
        transcript += str(transcript_segment["text"]).encode("utf-8").decode("utf-8")
        audio_segment_file.close()
        os.remove(audio_segment_path)
    os.remove(audio_file)
    transcript = punctuation_assistant(transcript)['choices'][0]['message']['content']
    transcript = "全文总结：\n" + summary_assistant(transcript)['choices'][0]['message']['content'] + "\n原文：\n" + transcript
    print(f"transcript: {transcript}")
    return transcript


def milliseconds_until_sound(sound, silence_threshold_in_decibels=-20.0, chunk_size=10):
    trim_ms = 0  # ms
    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold_in_decibels and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms


def punctuation_assistant(ascii_transcript: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Your job is to adds punctuation to text. "
                           "What you are going to do should follow the rules below: \n"
                           "1. You have received a text which is transcribed from an audio file which we call it \"original text\".\n"
                           "2. The response should be presented in the language of the original text.\n"
                           "3. I need you to convert the original text into a new context. During "
                           "this process, please preserve the original words of the given original text and only insert "
                           "necessary punctuation such as periods, commas, capialization, symbols like dollar signs or "
                           "percentage signs, and formatting according to the language of the provided original text. And I "
                           "hope you to separate the original text into several paragraphs based on the meaning. Please "
                           "use only the provided original text. \n"
            },
            {
                "role": "user",
                "content": ascii_transcript
            }
        ]
    )
    return response


def summary_assistant(ascii_transcript: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Your job is to summarize text. "
                           "What you are going to do should follow the rules below: \n"
                           "1. You have received a text which we call it \"original text\".\n"
                           "2. The response should be presented in the language of the original text.\n"
                           "3. I need you to make a brief statement of the main points of the original text."
                           "Please use only the provided original text. \n"
            },
            {
                "role": "user",
                "content": ascii_transcript
            }
        ]
    )
    return response
