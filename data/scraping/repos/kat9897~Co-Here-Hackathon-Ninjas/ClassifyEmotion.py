import _csv
import math
import re
import gzip
import json
import numpy as np
import cohere
import nltk

nltk.download('punkt')

from cohere.classify import Example

co = cohere.Client('Odj4ClpHI3fobuB9bErJkfzlszJwQ018Z99gUAyE')
KEY = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


def classify_chunks(response, data=True, if_print=False):
    results = {KEY[i]: [] for i in range(len(KEY))}
    for i in range(len(response.classifications)):
        prediction = int(response.classifications[i].prediction)
        confidence = response.classifications[i].confidence
        if confidence >= 0.5:
            results[KEY[prediction]].append(confidence)
        if data:
            data = response.classifications[i].input
            if if_print:
                print(f'Input: {data}')
        if if_print:
            print(f'Class: {prediction}, Confidence: {confidence} \n')
    return results


def feedback(results, if_print=False):
    if if_print:
        print(results, '\n')
    evaluation = []
    for i in range(len(KEY)):
        data = np.array(results[KEY[i]])
        if len(data) == 0:
            data = [0]
        average = np.mean(data)
        n = len(data)
        coefficient = n / (2 + n)
        evaluation.append(coefficient * average * max(data))
    if if_print:
        print(evaluation)
    return KEY[np.argmax(evaluation)], 2*np.max(evaluation)


def split_input(user_input):
    sentences = nltk.sent_tokenize(user_input)
    return sentences


def split_input_self(user_input):
    Replace = [' and ', ' that ', ' thus ', ' then ', ' however ', ' but ']
    for i in range(len(Replace)):
        user_input = user_input.replace(Replace[i], '.')
    cut_user_prompt = re.split(r'[,.!?-]', user_prompt)
    cut_user_prompt = list(filter(None, cut_user_prompt))
    return cut_user_prompt


def classify_emotion(user_prompt):
    user_chunks = split_input(user_prompt.strip())

    response = co.classify(
        model='0c667180-ef42-4c4d-aee8-e6fa29177e43-ft',
        inputs=user_chunks)

    results = classify_chunks(response)
    emotion, confidence = feedback(results)
    return emotion, confidence
