import tiktoken
import pandas as pd
import os
import openai
import time
import glob
import shutil
from dotenv import load_dotenv
from multiprocessing import Pool

# Settings
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
SCRAPE_FOLDER = 'data/processed/scraped'
PROCESSED_FOLDER = 'data/processed/processed_scraped'
EMBEDDING_FOLDER = 'data/processed/embeddings'
TOKENIZER_NAME = "cl100k_base"
MAX_TOKENS = 500
MAX_RETRIES = 3
SLEEP_TIME = 60
POOL_SIZE = 10

# Load the tokenizer
tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)


def create_embeddings(text, index):
    print(f"Creating embeddings for chunk {index + 1}")
    embedding = openai.Embedding.create(input=text, engine='text-embedding-ada-002')['data'][0]['embedding']
    return embedding


def split_into_many(text, max_tokens=MAX_TOKENS):
    sentences = text.split('. ')
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0
        if token > max_tokens:
            continue
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


def process_file(file):
    domain = os.path.basename(file).replace('_scraped.csv', '')
    print(f"Processing {domain}...")

    df = pd.read_csv(file, index_col=0)
    df.columns = ['title', 'text']
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()

    shortened = []
    for row in df.iterrows():
        if row[1]['text'] is None:
            continue
        if row[1]['n_tokens'] > MAX_TOKENS:
            shortened += split_into_many(row[1]['text'])
        else:
            shortened.append(row[1]['text'])

    df_shortened = pd.DataFrame(shortened, columns=['text'])
    df_shortened['n_tokens'] = df_shortened.text.apply(lambda x: len(tokenizer.encode(x)))
    df_shortened.n_tokens.hist()

    print(f"Total chunks to be processed: {len(df_shortened)}")
    print("Creating embeddings...")

    df_shortened['embeddings'] = None
    index = 0
    while index < len(df_shortened):
        for i in range(MAX_RETRIES):
            try:
                print(f"Processing chunk {index + 1}, attempt {i + 1}...")
                df_shortened.at[index, 'embeddings'] = create_embeddings(df_shortened.at[index, 'text'], index)
                index += 1
                break
            except openai.error.APIError as e:
                print(f"Error: {e}")
                if i < MAX_RETRIES - 1:
                    print(f"Retrying in 60 seconds...")
                    time.sleep(SLEEP_TIME)
                else:
                    raise Exception("Failed to create embeddings after 3 attempts")

    if not os.path.exists(EMBEDDING_FOLDER):
        os.makedirs(EMBEDDING_FOLDER)

    output_file = os.path.join(EMBEDDING_FOLDER, f"{domain}_embeddings.csv")
    df_shortened.to_csv(output_file, index=False)

    print("Done!")
    df_shortened.head()

    # Move processed file to processed folder
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)

    shutil.move(file, PROCESSED_FOLDER)


def main():
    pool = Pool(POOL_SIZE)  # Create a multiprocessing Pool
    while True:
        csv_files = glob.glob(f'{SCRAPE_FOLDER}/*.csv')
        if not csv_files:
            print("No files found, checking again in 60 seconds.")
            time.sleep(SLEEP_TIME)
            continue

        pool.map(process_file, csv_files)  # process files concurrently

        print("All files processed. Checking again in 60 seconds.")
        time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    main()
