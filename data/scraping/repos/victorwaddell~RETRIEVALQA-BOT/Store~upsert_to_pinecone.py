import time
from os.path import basename, splitext
from uuid import uuid4

import openai
from tqdm.auto import tqdm

from Split.tokenizer_split import token_splitter 


def create_embeddings(texts, embed_model):  # Creates embeddings for texts
    try:  # Tries to create embeddings
        return openai.Embedding.create(input = texts, engine = embed_model)
    except Exception as e:  # Prints error if embedding fails
        print(f"Rate limit or other exception hit during embedding: {e}")
        time.sleep(5)
        return create_embeddings(texts, embed_model)  # Retries embedding

def upsert_data_to_index(files, embed, index):  # Upserts data to Pinecone index
    batch_limit = 100
    texts = []
    metadatas = []

    for i, document in enumerate(tqdm(files)): 
        title = document.metadata.get('title', None)
        if not title:
            source_path = document.metadata['source']
            title = splitext(basename(source_path))[0] if source_path else 'Untitled Document'
        
        metadata = {'id': str(document.metadata['source'] + f'-{i}'),
                    'source': document.metadata['source'],
                    'title': title}
        
        document_texts = token_splitter.split_text(document.page_content)  # Now we create chunks from the document text
        document_metadatas = [{"chunk": j, "text": text, **metadata} for j, text in enumerate(document_texts)] # create individual metadata dicts for each chunk
        texts.extend(document_texts)  # append these to current batches
        metadatas.extend(document_metadatas)
        if len(texts) >= batch_limit:  # if we have reached the batch_limit we can add texts
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors = zip(ids, embeds, metadatas))
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors = zip(ids, embeds, metadatas))
    print("Upserting splits to Pinecone index complete!\n")
