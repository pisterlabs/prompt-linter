import faiss
import csv
import openai
import numpy as np
import argparse
import dotenv
import os

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class VectorSearch:
    def __init__(self, index_filename, chunk_filename):
        self.client = openai.OpenAI()

        self.index = faiss.read_index(index_filename)

        with open(chunk_filename, 'r', newline='', encoding='utf-8') as inputfile:
            reader = csv.DictReader(inputfile)
            self.idx_to_chunk = {int(row['numeric_id']): row['text_chunk'] for row in reader}
        
    def search(self, query_text, limit=1):
        query_embedding = [self.client.embeddings.create(input=[query_text], model="text-embedding-ada-002").data[0].embedding]
        D, I = self.index.search(np.array(query_embedding), limit)
        return [self.idx_to_chunk[idx] for idx in I[0]]

def main():
    parser = argparse.ArgumentParser(description="Add embeddings of text chunks to CSV file")
    parser.add_argument("index_file", help="Path of FAISS index file")
    parser.add_argument("chunk_file", help="Path of the text chunks file")
    args = parser.parse_args() 
    search = VectorSearch(args.index_file, args.chunk_file)

    while True:
        query_text = input("Enter a query: ")
        results = search.search(query_text)
        print(f"Closest results: {results}")

if __name__ == "__main__":
    main()