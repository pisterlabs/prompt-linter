import openai
import pinecone
import pandas as pd
import logging
import os
import itertools
import time

from logfmter import Logfmter
from dotenv import load_dotenv

import tiktoken

formatter = Logfmter(keys=["ts", "level"], mapping={"ts": "asctime", "level": "levelname"})
logging.getLogger().setLevel(logging.INFO)

load_dotenv()

conn_string = os.getenv('db_connect_string')
pinecone_api_key = os.getenv('pinecone_api_key')
pinecone_environment = os.getenv('pinecone_env')
openai.api_key = os.getenv('openai_api_key')
model = "text-embedding-ada-002"
enc = tiktoken.get_encoding("cl100k_base")

pinecone.init(api_key=pinecone_api_key,
              environment=pinecone_environment)
index = pinecone.Index('moneymaker')

def stock_universe_pinecone_import():
    try:
        # check if 'moneymaker' index already exists (only create index if not)
        if 'moneymaker' not in pinecone.list_indexes():
            pinecone.create_index('moneymaker', dimension=1536, metric='dotproduct')

        tickers_list = pd.read_sql_table("stocks", conn_string, schema=None, index_col=None, coerce_float=True,
                                         parse_dates=None,
                                         columns=None, chunksize=None).to_dict(orient='records')
        index_id = 0
        start = time.time()
        for tickers_chunk in chunks(tickers_list, batch_size=50):
            vector_list = []
            embeddings = get_news_embeddings_dense(tickers_chunk)
            chunk_index = 0
            for ticker_details in tickers_chunk:
                meta = {"ticker": ticker_details['ticker'],
                        "rsi_rating": ticker_details['rsi_rating'],
                        "sma_rating": ticker_details['sma_rating'],
                        "macd_rating": ticker_details['macd_rating']}
                vector_list.append(tuple([str(index_id), embeddings['data'][chunk_index]['embedding'], meta]))
                index_id = index_id + 1
                chunk_index = chunk_index + 1
            logging.info(f"OpenAI embedding batch processed: {index_id} of {len(tickers_list)} tickers processed")
            index.upsert(vectors=vector_list)
            logging.info(f"Pinecone load batch processed {index_id} of {len(tickers_list)} tickers processed")
        end = time.time()
        logging.info(f'stock universe records loaded from DB to pinecone in {end - start} seconds')
    except BaseException as be:
        logging.error(be)
        raise be


def get_news_embeddings_dense(tickers_list):
    embedding_list = []
    for ticker_details in tickers_list:
        beta_rating = "med"
        pe_rating = "med"
        try:
            if float(ticker_details['beta']) <= 1.0:
                beta_rating = "low"
            elif float(ticker_details['beta']) >= 1.5:
                beta_rating = "high"
        except ValueError:
            logging.error(f"no beta rating for {ticker_details['ticker']}")
        try:
            if int(ticker_details['pe']) <= 10:
                pe_rating = "low"
            elif int(ticker_details['pe']) >= 30:
                pe_rating = "high"
        except ValueError:
            logging.error(f"no PE rating for {ticker_details['ticker']}")

        news_tags = f"ticker {ticker_details['ticker']} name {ticker_details['name'].lower()} industry {ticker_details['industry'].lower()} sector {ticker_details['sector'].lower()} beta {beta_rating} pe {pe_rating}"

        news_tokens = enc.encode(news_tags + "\n" + ticker_details['news'])[:8191]
        embedding_list.append(news_tokens)
    logging.info(f"encoding complete for: {tickers_list[0]['ticker']} to {tickers_list[-1]['ticker']}")
    openai_embeddings = openai.Embedding.create(input=embedding_list, engine=model)

    return openai_embeddings

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def pinecone_query(prompt, max_rsi_rating, max_macd_rating, max_sma_rating):
    xq = openai.Embedding.create(input=prompt, engine=model)['data'][0]['embedding']
    res = index.query([xq], top_k=10, include_metadata=True,
                      filter={"rsi_rating": {"$lte": max_rsi_rating},
                              "macd_rating": {"$lte": max_macd_rating},
                              "sma_rating": {"$lte": max_sma_rating},
                              })
    values = []
    for match in res['matches']:
        value = {
            "score": f"{match['score']}",
            "ticker": match['metadata']['ticker'],
            "rsi_rating": match['metadata']['rsi_rating'],
            "macd_rating": match['metadata']['macd_rating'],
            "sma_rating": match['metadata']['sma_rating'],
        }
        values.append(value)

    return values
