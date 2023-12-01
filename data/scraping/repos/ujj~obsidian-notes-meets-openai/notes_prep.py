import os
import pandas as pd
from transformers import GPT2TokenizerFast
import openai
import time
from openai.embeddings_utils import get_embedding

notes_files = []

openai.api_key = "YOUR_OPEN_AI_API_KEY"


def fetch_embeddings(item, engine):
    embedding = get_embedding(item, engine)
    time.sleep(1)
    return embedding


def load_dataset():
    print("\n\nLoading the notes data in a dataframe\n\n")
    df = pd.read_csv('ujjwal_notes.csv')
    df = df[['Note', 'Title']]
    df = df.dropna()
    df['combined'] = "Title: " + df.Title.str.strip() + \
        "; Content: " + df.Note.str.strip()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    df['n_tokens'] = df.combined.apply(lambda x: len(tokenizer.encode(x)))
    df = df[df.n_tokens < 2000].tail(1_000)
    return(df)


def save_embeddings(df):
    print("\n\nRunning babbage similarity and search embeddings\n\n")
    df['babbage_similarity'] = df.combined.apply(
        lambda x:
        fetch_embeddings(x, engine='text-similarity-babbage-001'))
    df['babbage_search'] = df.combined.apply(
        lambda x:
        fetch_embeddings(x, engine='text-search-babbage-doc-001'))
    df.to_csv('ujjwal_notes_embeddings.csv')


def build_csv_file(notes_files):
    df = pd.DataFrame()
    for filepath in notes_files:
        with open(filepath, 'r') as fopen:
            note = fopen.read()
            print(filepath)
            note_df = pd.DataFrame(
                [[note, os.path.basename(filepath)]], columns=["Note", "Title"]
                )
            df = pd.concat([df, note_df], axis=0)
    df.to_csv("ujjwal_notes.csv")


def extract_note_paths(dir_path):
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            notes_files.append(os.path.join(dir_path, path))
        else:
            extract_note_paths(os.path.join(dir_path, path))
    return notes_files


if __name__ == "__main__":
    notes_files = extract_note_paths("/YOUR/PATH/TO/NOTES")
    build_csv_file(notes_files)
    df = load_dataset()
    save_embeddings(df)
