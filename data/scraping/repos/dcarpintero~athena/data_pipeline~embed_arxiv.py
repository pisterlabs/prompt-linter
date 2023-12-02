import requests
import cohere
import json
import time
import logging
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

BATCH_SIZE = 500
ARXIV_JSON = "data/arxiv_cs.CL.json"
ARXIV_EMBEDDINGS_JSONL = "data/arxiv_cs.CL_embedv3.jsonl"

def load_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"An error occurred while loading JSON: {e}")
        return []


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
def embed(co_client, texts):
    return co_client.embed(
            model='embed-english-v3.0',
            texts=texts,
            input_type='search_document',
            ).embeddings


def embed_batch(batch, co_client, file):
    titles = [paper['title'] for paper in batch]
    summaries = [paper['summary'] for paper in batch]

    title_embeddings = embed(co_client, titles)
    summary_embeddings = embed(co_client, summaries)

    for i, paper in enumerate(batch):
        paper['embeddings'] = {
            'title': title_embeddings[i],
            'summary': summary_embeddings[i]
        }
        json.dump(paper, file)
        file.write('\n')


def process_embeddings_and_save(data, co_client, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for i in range(0, len(data), BATCH_SIZE):
                batch = data[i:i + BATCH_SIZE]
                logging.info(f"embeddings_batch: {i} to {i + BATCH_SIZE}")
                embed_batch(batch, co_client, file)
                time.sleep(5)
    except Exception as e:
        logging.error(f"Error saving to JSONL: {e}")


def load_environment_vars():
    logging.info("Loading environment variables")
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise EnvironmentError("COHERE_API_KEY environment variable not set.")
    return cohere_api_key

@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
def initialize_cohere_client(api_key):
    logging.info("Initializing Cohere client")
    return cohere.Client(api_key)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data = load_json(ARXIV_JSON)
    if not data:
        logging.error("Failed to load data from JSON.")
        return

    api_key = load_environment_vars()
    cohere_client = initialize_cohere_client(api_key)

    process_embeddings_and_save(data, cohere_client, ARXIV_EMBEDDINGS_JSONL)

    logging.info("Processing completed and saved.")

if __name__ == "__main__":
    main()
