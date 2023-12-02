# https://github.com/openai/openai-cookbook/tree/main/examples/vector_databases/weaviate
# micromamba install datasets apache-beam
# pip install datasets apache-beam weaviate-client
#
# micromamba create -n test00
# micromamba install -n test00 -c conda-forge cython matplotlib h5py pillow protobuf scipy requests tqdm flask ipython openai python-dotenv tiktoken lxml tqdm python-magic datasets apache-beam

import os
import openai
import dotenv
import weaviate
from datasets import load_dataset

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


tmp0 = weaviate.auth.AuthApiKey(os.environ['WEAVIATE_API_KEY'])
tmp1 = {"X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY']} #optional
client = weaviate.Client(url=os.environ['WEAVIATE_API_URL'], auth_client_secret=tmp0, additional_headers=tmp1)


# Define the Schema object to use `text-embedding-ada-002` on `title` and `content`, but skip it for `url`
if 'Article' in {x['class'] for x in client.schema.get()['classes']}:
    client.schema.delete_class("Article")
article_schema = {
    "class": "Article",
    "description": "A collection of articles",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text"
        }
    },
    "properties": [{
        "name": "title",
        "description": "Title of the article",
        "dataType": ["string"]
    },
    {
        "name": "content",
        "description": "Contents of the article",
        "dataType": ["text"]
    },
    {
        "name": "url",
        "description": "URL to the article",
        "dataType": ["string"],
        "moduleConfig": { "text2vec-openai": { "skip": True } }
    }]
}
client.schema.create_class(article_schema)

dataset = list(load_dataset("wikipedia", "20220301.simple")["train"])[:25]

client.batch.configure(
    batch_size=10,
    dynamic=True,
    timeout_retries=3,
#   callback=None,
)
