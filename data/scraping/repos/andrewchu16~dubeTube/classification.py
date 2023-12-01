from email.mime import audio
import os
import cohere
import speech_recognition

from dotenv import load_dotenv
from cohere.classify import Example
from moviepy.editor import VideoFileClip

# change the file path later
AUDIO_WAV = "./classification/audio.wav"

load_dotenv()
API_KEY = os.getenv("API_KEY")

BASE_TAG_ACCEPTANCE = 0.2
PROMPT_FILE = "./classification/examples.txt"
SUMMARY_FILE = "./classification/summary.txt"

def classify(file_path):

    video = VideoFileClip(file_path)
    video.audio.write_audiofile(AUDIO_WAV)
    transcript = transcribe_audio(AUDIO_WAV)

    co = cohere.Client(API_KEY)
    summary_prompt = "Summarize this:\n"
    with open(SUMMARY_FILE, "r") as summary_file:
        summary_prompt += "".join(summary_file.readlines())
    summary_prompt += "Transcript: " + transcript.rstrip() + "\nSummary: "

    # if temperature goes any higher or lower than 0.5, it doesn't work as well
    summary_generation = co.generate(
        model = 'xlarge',
        prompt = summary_prompt,
        stop_sequences = ["--"],
        max_tokens = 50,
        num_generations = 1,
        temperature = 0.5
    )
    print(summary_generation)
    summary = summary_generation.generations[0].text.rstrip("--")
    
    if (transcript == ""):
        return ["No Audio"]

    prompts = []
    with open(PROMPT_FILE, "r") as prompt_file:
        lines = prompt_file.readlines()
        for i in range(0, len(lines), 2):
            prompts.append(Example(lines[i].rstrip(), lines[i+1].rstrip()))

    labels = []
    classifications = co.classify(
        model = "medium",
        inputs=[transcript],
        examples=prompts
    )
    
    for item in classifications.classifications[0].confidence:
        if item.confidence >= BASE_TAG_ACCEPTANCE:
            labels.append(item.label)

    return transcript, summary, labels



def transcribe_audio(file_path):
    with speech_recognition.AudioFile(file_path) as audio_file:
        recognizer = speech_recognition.Recognizer()
        audio_data = recognizer.record(audio_file)
        try:
            auto_text = recognizer.recognize_google(audio_data)
        except:
            # there is no audio
            return ""
        return auto_text
