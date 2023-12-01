import os
from typing import List
import cohere
from cohere.classify import Example
from sklearn.metrics import f1_score
import pandas as pd
import os

API_KEY = '[COHERE_API_KEY]'


def load_train_data(language: str, use_article_text: bool):
    assert language in get_available_languages(), 'Requested language is not available'
    df = pd.read_csv(f'../../data/{language}/train.tsv',
                     sep='\t', usecols=['category', 'headline', 'text'])
    if use_article_text:
        df['headline'] = df['headline'] + '\n' + df['text']
    return df.drop(columns='text')


def load_validation_data(language: str, use_article_text: bool):
    assert language in get_available_languages(), 'Requested language is not available'
    df = pd.read_csv(f'../../data/{language}/dev.tsv', sep='\t',
                     usecols=['category', 'headline', 'text'])
    if use_article_text:
        df['headline'] = df['headline'] + '\n' + df['text']
    return df.drop(columns='text')


def load_test_data(language: str, use_article_text: bool):
    assert language in get_available_languages(), 'Requested language is not available'
    df = pd.read_csv(f'../../data/{language}/test.tsv', sep='\t',
                     usecols=['category', 'headline', 'text'])
    if use_article_text:
        df['headline'] = df['headline'] + '\n' + df['text']
    return df['headline'].tolist(), df['category'].tolist()


def get_available_languages():
    return sorted(list(os.listdir('../../data')))


def get_samples_per_class(df: pd.DataFrame, n: int):
    return df.groupby('category').apply(lambda cat: cat.sample(n, random_state=42))


def train_data_to_cohere_examples(df: pd.DataFrame):
    examples = []
    for row in df.itertuples():
        examples.append(Example(row.headline, row.category))
    return examples


def get_classifications(co, examples: List[Example], inputs: List[str]):
    response = co.classify(
        model='multilingual-22-12',
        inputs=inputs,
        examples=examples
    )
    return response


def process_classifications(inputs, ground_truth, classifications):
    data, accuracy = [], 0
    f1_class = [c.prediction for c in classifications]
    f1 = f1_score(ground_truth, f1_class, average='weighted')
    for input_text, gt, classification in zip(inputs, ground_truth, classifications):
        accuracy += 1 if gt == classification.prediction else 0
        data.append({
            'text': input_text,
            'ground_truth': gt,
            'prediction': classification.prediction,
            'confidence': classification.confidence,
            'labels': classification.labels
        })
    return data, accuracy / len(inputs), f1


def add_metadata(metadata, data):
    return {**metadata, 'data': data}


def get_co():
    return cohere.Client(API_KEY)
