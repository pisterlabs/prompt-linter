#%% 1. Initializing the Hugging Face Embedding Pipeline

#! Initializing the embedding pipeline that will handle the transformation of our docs into vector embeddings

from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)



#%% 2. Building the Vector Index
import os
import pinecone
import time
from datasets import load_dataset # jamescalam/llama-2-arxiv-papers-chunked for training

# get API key from app.pinecone.io and environment from console
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY',
    environment=os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENVIRONMENT'
)


index_name = 'llama-2-rag'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

"""Now we connect to the index:"""

index = pinecone.Index(index_name)
index.describe_index_stats()



data = load_dataset('jamescalam/llama-2-arxiv-papers-chunked', split='train').to_pandas()

batch_size = 32

for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
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

#%% 3. Initializing the Hugging Face Pipeline
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline, BitsAndBytesConfig
import torch
from langchain.llms import HuggingFacePipeline

# The first thing we need to do is initialize a `text-generation` pipeline with Hugging Face transformers. The Pipeline requires three things that we must initialize first, those are:
# A LLM, in this case it will be `gpt2`.
# The respective tokenizer for the model.

model_id = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = GPT2LMHeadModel.from_pretrained(model_id,
                                        device_map='auto',
                                        offload_folder='archive',
                                        low_cpu_mem_usage=True)

generate_text = pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
    pad_token_id=tokenizer.eos_token_id
)



# Now to implement this in LangChain

llm = HuggingFacePipeline(pipeline=generate_text)

#%% 4. Initializing a RetrievalQA Chain
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

text_field = 'text'  # field in metadata that contains text content

vectorstore = Pinecone(index,
                       embed_model.embed_query,
                       text_field)





rag_pipeline = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=vectorstore.as_retriever())

rag_pipeline('how does the performance of llama 2 compare to other local LLMs?')

