import sys
import os
import argparse
from dotenv import load_dotenv
import openai
import csv

MODEL = "text-embedding-ada-002"
OUTPUT_DIRECTORY = "data/embeddings"

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parser = argparse.ArgumentParser(description='Embed documents into a csv file')
    parser.add_argument('--docs_dir', type=str, default="", help='dir of docs to embed')
    args = parser.parse_args()
    if args.docs_dir == "":
        print("ERROR: Docs dir is required.")
        sys.exit(1)

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    csvfile = open(OUTPUT_DIRECTORY + '/' + args.docs_dir.replace("/", "_") + '.csv', 'w', newline='')
    fieldnames = ['file_name', 'content', 'total_tokens', 'embedding']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in os.listdir(args.docs_dir):
        f = os.path.join(args.docs_dir, filename)
        if not os.path.isfile(f):
            continue

        print("Embedding " + f)
        content = open(f, "r").read()
        resp = openai.Embedding.create(input=content, model=MODEL)
        embedding = resp['data'][0]['embedding']
        total_tokens = resp['usage']['total_tokens']

        writer.writerow({'file_name': f,
                         'content': content,
                         'total_tokens': total_tokens,
                         'embedding': embedding})

    print("Done!")

if __name__ == "__main__":
    main()
