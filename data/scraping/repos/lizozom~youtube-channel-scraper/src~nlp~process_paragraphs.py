import time
import os
import re
import json
import requests
import openai
from openai.error import RateLimitError
import warnings
from dotenv import load_dotenv

# from transformers import pipeline
import numpy as np
from src.nlp.utils import get_scores_folder, get_processed_caption_folder, CHANNEL

load_dotenv()

API_URL = os.environ.get("HUGGINGFACE_API_URL")
BEARER = os.environ.get("HUGGINGFACE_API_KEY")
print(BEARER)
HEADERS = {"Authorization": f"Bearer {BEARER}"}
openai.api_key = os.environ.get("OPENAI_API_KEY")

HF_CANDIDATE_LABELS = ['story', 'recipe']
ARR_TITLES = ['index', 'recipe_score', 'history_score', 'summary']


INSTRUCTION = """
Given the paragraph below, create A JSON that represent the summary of the paragraph and the probability of it being about recipes and history.
The json object should have these keys enclosed in double quotes: "summary", "recipe and cooking" and "history and stories".
The json value for summary should be a one sentence summary of the topic of the paragraph.
The json values for "recipe and cooking" and "history and stories" should be zero shot classification probabilities.
The resulting JSON object should be in this format: {"summary": "string", "recipe and cooking":number, "history and stories":number}.
Don't add any text after the json.
"""
# par_transcript_2022-03-22T15:00:07Z_JbmHZbTpoDY.txt
PROCESS_VIDEO_IDS = ['JbmHZbTpoDY']#'1MAB-VVqjOE', '5S7Bb0Qg-oE', 'KTVPV-15GL0', 'ry5Du60WPGU', 'h6XvMKdD2tY', 'oqQzWg9pXmg', '7hYBesohRK0', 'oPTdSMOQRnY']


def query_huggingface(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
    return response.json()


def load_script_paragraphs(file_path) -> list[str]:
    with open(os.path.join(file_path), 'r', encoding='utf-8') as f:
        paragraphs = f.read().split('\n\n')
        print(f"Loaded {len(paragraphs)} paragraphs")
        return paragraphs

def classify_transcript_paragraphs_hf(paragraphs: list[str]):
    results = []
    for i, par in enumerate(paragraphs):
        print(f"Processing paragraph {i}")
        result = query_huggingface({
            "inputs": par,
            "parameters": {
                "candidate_labels":', '.join(HF_CANDIDATE_LABELS),
                "multi_label": True
            },
        })
        print(result)
        scores = result["scores"]
        labels = result["labels"]
        score_map = dict(zip(labels, scores))
        results.append([i, score_map['story'], score_map['recipe']])
    return results


def classify_transcript_paragraphs_gpt(paragraphs: list[str]):
    results = []
    for i, par in enumerate(paragraphs):
        print(f"Processing paragraph {i}")
        
        chat_history = [
            {"role": "system", "content": "You are an tool specialized in zero shot classification."},
            {"role": "user", "content": INSTRUCTION},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": par}
        ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=chat_history
            )
        except RateLimitError:
            print("Got a rate limit error. Sleeping.")
            time.sleep(30)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=chat_history
            )
        reason = response['choices'][0]['finish_reason']
        message = response['choices'][0]['message']['content']
        print(message)
        parsedMsg = json.loads(message)

        print(f"Received result for piece {i}")
        if reason != 'stop':
            warnings.warn(f"Finished with code {reason} for")

        results.append([i, parsedMsg['recipe and cooking'], parsedMsg['history and stories'], parsedMsg['summary']])
        time.sleep(1)

    return results

def process_transcripts():
    file_path = get_processed_caption_folder(CHANNEL)
    target_path = get_scores_folder(CHANNEL)
    transcript_files = os.listdir(file_path)
    for script_file in transcript_files:
        vid_id = None
        target_file = os.path.join(target_path, script_file.replace('.txt', '.csv'))
        if os.path.exists(target_file):
            print(f"Skipping {script_file} as it has already been processed.")
            continue
        try:
            pattern = r'(?<=_)[\w-]{11}(?=\.txt$)'
            match = re.search(pattern, script_file)
            if match:
                vid_id = match.group()
                print("YouTube Video ID:", vid_id)
            else:
                print(f"No YouTube Video ID found in {script_file}.")
        except TypeError:
            pass
        # if vid_id not in PROCESS_VIDEO_IDS:
        #     continue
        print("Processing file: ", script_file)
        paragraphs = load_script_paragraphs(os.path.join(file_path, script_file))
        res = classify_transcript_paragraphs_gpt(paragraphs)
        print(res)
        a = np.asarray(res, dtype=object)
        np.savetxt(target_file, a, delimiter=",", fmt=['%d', '%.5f', '%.5f',  '%s',], header=','.join(ARR_TITLES))
        time.sleep(5)

if __name__ == "__main__":
    process_transcripts()