# Get OpenAI embedding for a file
# pip install openai
# pip install "openai[embeddings]" (if using --embedding-utils)
# python embed.py <file>

import argparse
import openai
from openai.embeddings_utils import get_embedding

# Read in a file from command line argument

parser = argparse.ArgumentParser(description="Get OpenAI embedding for a file")
parser.add_argument("file", metavar="file", type=str, help="file to embed")
# Flag for encoding_format="float"
parser.add_argument(
    "--float",
    dest="encoding_format",
    action="store_const",
    const="float",
    help="use float encoding format vs. default base64",
)
# Flag to use embedding_utils
parser.add_argument(
    "--embedding-utils",
    dest="embedding_utils",
    action="store_const",
    const="embedding_utils",
    help="use embedding_utils to get embedding",
)
args = parser.parse_args()

with open(args.file, "r") as f:
    text = f.read()

    model_id = "text-embedding-ada-002"

    if args.embedding_utils:
        embedding = get_embedding(
            text=text, engine=model_id, encoding_format=args.encoding_format
        )
    else:
        embedding = openai.Embedding.create(
            input=text, model=model_id, encoding_format=args.encoding_format
        )["data"][0]["embedding"]
    print(embedding)
