import gzip
import json
import numpy as np
import cohere
from cohere.classify import Example
import os
from dotenv import load_dotenv

# Load in environment variable secrets
load_dotenv('.env')
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
COHERE_API_KEY = "UB5I1scUMGv1XHS8i9SISKx3xIpTTKZwyGGngndB"
co = cohere.Client(COHERE_API_KEY)

# Main function
def main():
    #text, label = read_gzip()
    #classify(text, label)
    emotion = "happy"
    therapist(emotion)

# Open dataset
def read_gzip():
    text = []
    label = []
    with gzip.open('venv/train.jsonl.gz', 'r') as fin:
        for line in fin:
            data = json.loads(line.decode('utf-8'))
            text.append(data["text"])
            label.append(data["label"])
    return text, label

# Classify all texts and labels using model large with cohere
def classify(text, label):
    examples = []
    for i in range(500):
        examples.append(Example(text[i], str(label[i])))
    inputs = ["I feel so energetic", "I want to die",
              'I\'m just so tired all the time.',
              'I\'m just so stressed all the time.',
              'I\'m just so worried all the time.',
              'I\'m just so angry all the time.',
              'I\'m just so frustrated all the time.']

    response = co.classify(
        model='large',
        inputs=inputs,
        examples=examples)

    for i in range(len(response.classifications)):
        prediction = response.classifications[i].prediction
        confidence = response.classifications[i].confidence
        inputs = response.classifications[i].input
        print(f'Input = {inputs}, Prediction = {prediction}, Confidence = {confidence}')

# Generic cohere generate api
def generate(text):
    prompt = f'Generate sentences based on the sentence {text[0]} tell me more'
    GENERATE = 1
    response = co.generate(
        model='xlarge',
        prompt=prompt,
        max_tokens=30,
        temperature=1,
        end_sequences=["--"]
    )

    for i in range(GENERATE):
        answer = response.generations[0].text
        print(answer)

# Generate therapist-like response to certain emotion
def therapist(text):
    prompt = f'Generate a therapist\'s helpful response and suggestions to a patient feeling the emotion {text}.'
    GENERATE = 1
    print(prompt)
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=300,
        temperature=0.9,
        end_sequences=["--"]
    )
    res = response.generations[0].text
    return res
    # for i in range(GENERATE):
    #     answer = response.generations[0].text
    #     print(answer)


def calculate_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == '__main__':
    main()
