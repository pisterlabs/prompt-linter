import sys
import os
import argparse
from dotenv import load_dotenv
import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity

MODEL = "text-embedding-ada-002"

def semantic_search(df, query, n=3):
    query_embedding = get_embedding(query, engine=MODEL)
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embedding))

    results = (df.sort_values("similarity", ascending=False).head(n))
    return results

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parser = argparse.ArgumentParser(description="Semantic search with embeddings")
    parser.add_argument('--embedding_csv', type=str, default="", help='embedding csv file')
    parser.add_argument('--query', type=str, default="", help='query to semantic search')
    args = parser.parse_args()
    if args.embedding_csv == "" or args.query == "":
        print("ERROR: Embedding CSV and query is required.")
        sys.exit(1)

    df = pd.read_csv(args.embedding_csv)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)

    results = semantic_search(df, args.query, n=3)
    print(results)

if __name__ == "__main__":
    main()
