import uuid
import dotenv
import sys
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

dotenv.load_dotenv()

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", default="all-mpnet-base-v2", help="Huggingface Hub model for embedding")

args = vars(parser.parse_args())

def createUUID():
    return str(uuid.uuid4())

## Main Function ##
def main(args):
    from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

    documents = json.load(sys.stdin)
    embeddings_model = HuggingFaceEmbeddings(model_name=args['model'])
    embeddings = embeddings_model.embed_documents([doc['text'] for doc in documents])

    docs = [{**doc,'embedding': embedding} for doc, embedding in zip(documents, embeddings)]
    json.dump(docs, sys.stdout, indent=2)

main(args)