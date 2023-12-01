import os
import sys
import requests
import openai
from pydub import AudioSegment
from rapidfuzz import fuzz
import string

# OpenAI Whisper API settings
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  
OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
MODEL_ID = "whisper-1"

# Deepgram API settings
DEEPGRAM_API_KEY = "YOUR_DEEPGRAM_API_KEY"  
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

def remove_punctuation(input_string):
    translator = str.maketrans('', '', string.punctuation)
    no_punct = input_string.translate(translator)
    return no_punct

def compare_strings(str1, str2):
    str1 = remove_punctuation(str1)
    str2 = remove_punctuation(str2)
    return fuzz.ratio(str1, str2)

def transcribe_openai(AUDIO_FILE_PATH):
    with open(AUDIO_FILE_PATH, "rb") as audio_file:
        response = openai.Audio.transcribe(MODEL_ID, audio_file)
    return response.text

def transcribe_deepgram(AUDIO_FILE_PATH):
    headers = {
        "Authorization": "Token " + DEEPGRAM_API_KEY,
        "Content-Type": "audio/mpeg"
    }
    with open(AUDIO_FILE_PATH, "rb") as audio_file:
        audio_data = audio_file.read()
    response = requests.post(DEEPGRAM_API_URL, headers=headers, data=audio_data)
    response.raise_for_status()
    return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]

def summarize_transcript(openai_transcript, GPT_MODEL):
    messages = [
        {
            "role": "system",
            "content": "You are a editor, writer, and stenographer. Summarize the provided transcription text. Be aware that some words may be incorrect or missing."
        },
        {
            "role": "user",
            "content": openai_transcript
        }
    ]

    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        max_tokens=100
    )
    return response.choices[0].message["content"]

def analyze_transcriptions(audio_content, openai_transcript, deepgram_transcript, GPT_MODEL):
    messages = [
        {
            "role": "system",
            "content": "You are a skilled editor, transcriber of speech, and stenographer. Your task is to review two transcripts of the same speech. Given context that explains the speech, provide a new corrected transcript that fixes the errors in each of the two original transcripts. Make sure to consider how words that sound similar can be mistranscribed. Use your knowledge of phonetics, pronunciation, speech patterns, modern slang, and more. Generate a highly accurate consensus transcript that preserves the original meaning and content as exactly as possible while fixing the errors. Be aware of different English dialects such as AAVE and do not correct grammar based on Standard American English. Censor inappropriate words with asterisks. Think step by step and be careful to maintain accuracy to the transcripts when possible. Do not hallucinate."
        },
        {
            "role": "user",
            "content": f"Here is a summary of the transcriptions:'{audio_content}'. Here are the two transcriptions which may have errors: OpenAI transcript: '{openai_transcript}', Deepgram transcript: '{deepgram_transcript}'. Provide a new corrected transcription that is faithful to the words used in the transcripts. Do not replace words with synonyms."
        }
    ]

    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages
    )
    return response.choices[0].message["content"]

if __name__ == "__main__":
    AUDIO_FILE_PATH = sys.argv[1]  
    GPT_MODEL = sys.argv[2]

    openai_transcript = transcribe_openai(AUDIO_FILE_PATH)
    deepgram_transcript = transcribe_deepgram(AUDIO_FILE_PATH)
    
    similarity = compare_strings(openai_transcript.lower(), deepgram_transcript.lower())
    print(f"The Levenshtein similarity between the two transcriptions is {similarity}%")

    audio_content = summarize_transcript(openai_transcript, GPT_MODEL)
    consensus_transcript = analyze_transcriptions(audio_content, openai_transcript, deepgram_transcript, GPT_MODEL)
    print("Consensus Transcript: ", consensus_transcript)
