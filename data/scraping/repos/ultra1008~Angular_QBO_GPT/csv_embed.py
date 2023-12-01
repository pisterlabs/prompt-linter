from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import json, os
from dotenv import load_dotenv

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')

def embeding(filename, filepath):
    loader = CSVLoader(file_path=filename, encoding="utf8")
    data = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)

    result = list()
    for t in data:
        query_result = embeddings.embed_query(t.page_content)
        result.append(query_result)
    with open('./JSON/vector-{}.json'.format(filepath), 'w') as outfile:
        json.dump(result, outfile, indent=2)

if __name__ == '__main__':
    embeding("./CSV/index.csv")
