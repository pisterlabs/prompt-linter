#%% 1.Import libraries
import os                                                           # Functions for interacting with the operating system
import pinecone                                                     # Vector database for similarity search and ranking
import pandas as pd                                                 # Data manipulation and analysis library
import time                                                         # Time-related functions
from torch import cuda                                              # PyTorch's CUDA library for GPU computations
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # Provides Hugging Face's transformer models for text embeddings
from dotenv import load_dotenv                                      # Reads .env files and sets environment variables

#%% 2.Set parameters and environment variables
load_dotenv()

# get API key from app.pinecone.io and environment from console
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY',
    environment=os.environ.get('PINECONE_ENV') or 'PINECONE_ENV'
)

# Initialize the HuggingFace Embedding model for indexing to Pinecone
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)

docs = [
    "this is one document",
    "and another document"
]

embeddings = embed_model.embed_documents(docs)

#%% 3.Create new database in Pinecone

index_name = 'duhocsinh-se'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pinecone.Index(index_name)

#%% 4.Load data and index to Pinecone
data = pd.read_json('crawled_data/universityadmissions.jsonl', lines=True)

batch_size = 16

for i in range(0, len(data), batch_size):
    print(i)
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"migrationsverket-{j}" for j, x in batch.iterrows()]
    texts = [x['chunk'] for i, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'text': x['chunk'],
         'source': x['source'],
         'title': x['title']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))
