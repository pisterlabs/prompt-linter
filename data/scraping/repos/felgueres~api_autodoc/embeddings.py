import os
import pandas as pd
import numpy as np
from constants import DTYPE_PDF
from typing import List
import openai
import time
import json

from db import read_from_db, write_many_to_db, write_to_db
from numpy import dot 
from numpy.linalg import norm

from extra.logger_config import setup_logger
logger = setup_logger(__name__)

def cos_sim(a,b): return dot(a, b)/(norm(a)*norm(b))

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def file_exists(file_path):
    return os.path.isfile(file_path)

def write_out(df, target_path):
    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path))
    df.to_csv(target_path, index=False, escapechar='\\')
    logger.info('Df write with success: {}'.format(file_exists(target_path)))
    return None

def pdf_to_df(fpath, source_id):
    from PyPDF2 import PdfReader
    with open(fpath, "rb") as f:
        reader = PdfReader(f)
        texts = []
        for i, page in enumerate(reader.pages): texts.append((i+1, page.extract_text()))
        df = pd.DataFrame(texts, columns=['page_number', 'text'])
        df['text'] = remove_newlines(df.text)
        df['source_id'] = source_id
        df.columns = ['page_number', 'text', 'source_id']
    return df

def split_into_many(page_number, text, tokenizer, max_tokens=500):
    words = text.split(' ')
    n_tokens = [len(tokenizer.encode(w)) for w in words]
    chunks = []
    chunk = []
    tokens_so_far = 0

    for w, tokens in zip(words, n_tokens):
        if tokens_so_far + tokens + 1 > max_tokens:
            chunks.append((page_number, " ".join(chunk)))
            chunk = []
            tokens_so_far = 0
        chunk.append(w)
        tokens_so_far += tokens 

    # recover last chunk
    if len(chunk) > 0:
        chunks.append((page_number, " ".join(chunk)))
    return chunks

def shorten_text(df, tokenizer, max_tokens=500):
    shortened = []
    for _, row in df.iterrows():
        if row['text'] is None:
            continue
        if row['n_tokens'] > max_tokens:
            print('Shortening page {}'.format(row['page_number']))
            shortened += split_into_many(row['page_number'], row['text'], tokenizer)
        else:
            print('Keeping page {}'.format(row['page_number']))
            shortened.append((row['page_number'],row['text']))
    return shortened

def prepare_df(df, max_tokens=500):
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    shortened_df = pd.DataFrame(shorten_text(df, tokenizer, max_tokens), columns=['page_number', 'text'])
    shortened_df['n_tokens'] = shortened_df.text.apply(lambda x: len(tokenizer.encode(x)))
    return shortened_df

def get_embeddings(df):
    df['embeddings'] = df.text.apply(lambda x: get_q_embeddings(x))
    return df

def pdf_to_embeddings(fname, user_id, source_id):
    fname = os.path.splitext(os.path.basename(fname))[0] 
    raw_path = f'users/{user_id}/files/{fname}.{DTYPE_PDF}'
    n_tokens = 0

    try:
        with open(raw_path, 'rb') as f:
            data = f.read()
            add_blob_q = '''INSERT INTO blobs (source_id,data,dtype) VALUES (?,?,?)'''

        blob_entry = (source_id, data, DTYPE_PDF)
        write_to_db(add_blob_q, blob_entry)

        df = pdf_to_df(raw_path, source_id)
        df = prepare_df(df)
        df = get_embeddings(df)
        df['embeddings'] = df.embeddings.apply(lambda x: str(x)) 
        df['source_id'] = source_id
        df['metadata'] = df.apply(lambda x: json.dumps({'page_number': x['page_number']}), axis=1) 
        df['title'] = fname 
        n_tokens = df.n_tokens.sum()
        entries = [(user_id, title, text, n_tokens, embeddings, source_id, metadata) for title, text, n_tokens, embeddings, source_id, metadata in df[['title', 'text', 'n_tokens', 'embeddings', 'source_id', 'metadata']].values]
        sql_insert_embeddings = '''INSERT INTO embeddings (user_id, title, text, n_tokens, embeddings, source_id, metadata) VALUES (?, ?, ?, ?, ?, ?,?) '''
        write_many_to_db(sql_insert_embeddings, entries)
        return {'status': 'success', 'n_tokens': n_tokens}

    except Exception as e:
        logger.info({e})
        return {'status': 'error', 'n_tokens': n_tokens }

def read_embeddings_from_db(sources_ids: List[str]):
    embeddings_sql = 'SELECT * FROM embeddings WHERE source_id IN (%s)' % ','.join('?' * len(sources_ids))
    data = read_from_db(embeddings_sql, [*sources_ids]) 
    df = pd.DataFrame(data)
    df['embeddings'] = df.embeddings.apply(eval).apply(np.array)
    return df

def compute_distances(data_df, question_df):
    data_df['distance'] = data_df.embeddings.apply(lambda d_e: cos_sim(question_df, d_e))
    return data_df

def get_q_embeddings(q, engine='text-embedding-ada-002', max_retries=5):
    openai.api_key = os.environ['OPENAI_KEY']
    retry_delay = 1
    backoff = 2
    # TODO: use backoff library 
    for i in range(max_retries):
        try:
            q_embeddings = openai.Embedding.create(input=q, engine=engine)['data'][0]['embedding']
            return q_embeddings
        except Exception as e:
            print(f"Error: {e}")
            if i == max_retries - 1: 
                return None
            time.sleep(retry_delay)
            retry_delay *= backoff

def fetch_passages(d_embeddings_df, max_passages=5, sort_by='distance', ascending=False):
    cur_len = 0
    sources = [] 
    for i, row in d_embeddings_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True).iterrows():
        cur_len += row['n_tokens'] + 4
        if i >= max_passages:
            break
        m = json.loads(row['metadata'])
        sources.append({'id': row['source_id'],
                        'title': row['title'], 
                        'text': row['text'], 
                        'score': 0, 
                        'n_tokens': row['n_tokens'], 
                        'page_number': m['page_number']})
    sources = sorted(sources, key=lambda x: x['page_number'], reverse=False)
    print('Pages: ', [s['page_number'] for s in sources])
    return sources 
