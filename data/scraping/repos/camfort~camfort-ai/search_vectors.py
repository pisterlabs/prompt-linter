import sqlite3
from openai.embeddings_utils import cosine_similarity
from openai.embeddings_utils import get_embedding
from openai import api_key
import pandas as pd
import argparse

defaultdbfile = 'vectors.db'
tabname = 'embeddings'

def main():
    global api_key
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', '-D', type=str, default=defaultdbfile, help='SQLite3 database filename for storing vectors')
    parser.add_argument('--api-key', type=str, default=None, help='OpenAI API key (or use env var OPENAI_API_KEY)')
    parser.add_argument('--num-results', '-n', type=int, default=3, help='Number of results to output')
    parser.add_argument('input', nargs=argparse.REMAINDER, type=str, help='Search terms')

    args = parser.parse_args()
    if args.api_key is not None:
        api_key = args.api_key

    searchtxt = ' '.join(args.input)

    con = sqlite3.connect(args.database)

    query = "SELECT path, name AS fname, firstLine, lastLine, vectorid, elem AS emb FROM embeddings JOIN vectors ON embeddings.vectorid = vectors.id ORDER BY vectorid, ord"
    df = pd.read_sql_query(query, con).groupby('vectorid').agg({'emb': list, 'path': 'first', 'fname': 'first', 'firstLine': 'first', 'lastLine': 'first'}).reset_index()

    search_functions(df, searchtxt, n=args.num_results)

def search_functions(df, code_query, n=3, pprint=True):
    embedding = get_embedding(code_query, engine='code-search-babbage-text-001')
    df['similarities'] = df.emb.apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False).head(n)
    if pprint:
        for r in res.iterrows():
            print(f'{str(round(r[1].similarities*100, 1))}%: {r[1].fname}: {r[1].path} line {r[1].firstLine}')

    return res


if __name__=="__main__":
  main()
