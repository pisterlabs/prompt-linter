from typing import Any, List, Dict, Union, Optional, Sequence
import argparse
import logging
import time
import faiss
import weaviate
import openai
import ray
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.node_parser.interface import NodeParser
from llama_index.schema import Document, BaseNode
from llama_index.vector_stores import WeaviateVectorStore
from semantic_search.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_DIM, DEFAULT_WEAVIATE_TEXT_KEY
from semantic_search.utils.common_utils import get_dir_size

logger = logging.getLogger('ray')

class NopNodeParser(NodeParser):
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        return nodes

def process_shard(embedding_model_id: str, shard: List[Document], batch_size:int, device:str) -> List[Document]:
    st = time.time()
    logger.info(f"Load embedding model {embedding_model_id}")
    embed_model = SentenceTransformer(embedding_model_id, device=device)
    logger.info(f"Starting process_shard of {len(shard)} chunks.")
    texts = [doc.text for doc in shard]
    # normalize embeddings so we can use a dot product to calculate cosine distance
    embeddings = embed_model.encode(texts, normalize_embeddings=True, batch_size=batch_size)

    for i in range(len(shard)):
        # convert np.ndarray to List[float], LlamaIndex does not like the Numpy array
        shard[i].embedding = embeddings[i].tolist() 
    et = time.time() - st
    logger.info(f"Shard completed in {et} seconds.")
    return shard


def main(num_shards:int, chunk_size:int, chunk_overlap:int, dataset_size:int, 
         faiss_index_path:str, min_doc_size:int, batch_size:int, weaviate_url:str,
         weaviate_index_name:str, num_gpus:float):

    if not faiss_index_path and not weaviate_url:
        raise ValueError('Must specify faiss_index_path or weaviate_url')

    if faiss_index_path and weaviate_url:
        raise ValueError('Must specify only one of faiss_index_path or weaviate_url')

    logger.info("Load dataset")

    # Note that there are full Wikipedia dumps in different languages
    # for dataset details see: https://huggingface.co/datasets/wikipedia
    dataset_name = 'wikipedia'
    dataset_data_dir = "20220301.simple"
    split = 'train' if dataset_size is None else f'train[:{dataset_size}]'
    dataset = load_dataset(dataset_name, dataset_data_dir, 
                           split=split, 
                           ignore_verifications=True, 
                           beam_runner='DirectRunner')
    logger.info(f"{dataset}")

    logger.info("Split documents")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EMBEDDING_MODEL)
    def tokenizer_len_fn(text):
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tokenizer_len_fn,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks:List[str] = []
    metadata:List[Dict] = []

    for record in dataset:
        # Ignore short documents
        if tokenizer_len_fn(record['text']) >= min_doc_size:
            meta = {
                'uri': record['url'],
                'title': record['title'],
                'source': f"{dataset_name}-{dataset_data_dir}"
            }
            # split current document
            record_texts = text_splitter.split_text(record['text'])
            # prepend title to chunks
            record_texts = [record['title'] + '\n' + text for text in record_texts]
            # clone document metadata for each chunk
            record_metadatas = [meta for _ in range(len(record_texts))]
            chunks.extend(record_texts)
            metadata.extend(record_metadatas)

    chunks = [Document(text=text, metadata=meta) for text, meta in zip(chunks, metadata)]
    logger.info(f"{len(chunks)} chunks")
    chunk_lengths = [tokenizer_len_fn(chunk.text) for chunk in chunks]
    logger.info(f'Chunk stats: min={int(np.min(chunk_lengths))} '
                f'mean={int(np.mean(chunk_lengths))} '
                f'max={int(np.max(chunk_lengths))}')

    logger.info(f"Calculate embeddings")
    shards = [shard.tolist() for shard in np.array_split(chunks, num_shards)]

    # Connect to Ray cluster
    logger.info('Connect to Ray cluster')
    ray.init(address='auto')

    if num_gpus > 0:
        num_cpus = 0
        device = 'cuda'
    else:
        num_cpus = 1
        num_gpus = 0
        device = 'cpu'

    # Submit tasks to calculate embeddings in parallel
    logger.info(f"Submit {num_shards} embedding tasks")
    st = time.time()
    futures = [ray.remote(
        num_cpus=num_cpus, 
        num_gpus=num_gpus)(process_shard).remote(
            DEFAULT_EMBEDDING_MODEL, 
            shards[i], 
            batch_size,
            device) for i in range(num_shards)]

    results:List[List[Document]] = ray.get(futures)
    et = time.time() - st
    logger.info(f"Embedding calculation finished in {et}s")

    # Flatten the results from the embedding tasks
    chunks = [item for sublist in results for item in sublist]

    # The LLM and embedding model components in the service context have OpenAI
    # default implementations that validate the existance of the API key. We do
    # not need these components but the service context will initialize them. The
    # fake key also raises an error in case OpenAI is called unexpectedly.
    openai.api_key = "sk-000000000000000000000000000000000000000000000000"

    if faiss_index_path:

        logger.info(f"Build FAISS index")

        faiss_index = faiss.IndexFlatIP(DEFAULT_EMBEDDING_DIM)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build a service context. Configure a NOP NodeParser because the
        # documents are already split up into chunks.
        service_context = ServiceContext.from_defaults(node_parser=NopNodeParser())

        # Index all documents
        index = VectorStoreIndex.from_documents(chunks, 
                                                storage_context=storage_context, 
                                                service_context=service_context)

        # Write index to disk
        logger.info(f'Write index to: {faiss_index_path}')
        index.storage_context.persist(faiss_index_path)
        index_size = get_dir_size(faiss_index_path)
        logger.info(f"Index size: {index_size} bytes")

    else:

        logger.info(f"Build Weaviate index {weaviate_index_name} @ {weaviate_url}")

        client = weaviate.Client(url=weaviate_url)

        # Delete index if it already exists
        client.schema.delete_class(weaviate_index_name)

        vector_store = WeaviateVectorStore(weaviate_client=client, 
                                           index_name=weaviate_index_name, 
                                           text_key=DEFAULT_WEAVIATE_TEXT_KEY)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build a service context. Configure a NOP NodeParser because the
        # documents are already split up into chunks.
        service_context = ServiceContext.from_defaults(node_parser=NopNodeParser())

        # Index all documents
        index = VectorStoreIndex.from_documents(chunks, 
                                                storage_context=storage_context, 
                                                service_context=service_context)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build vector store')
    parser.add_argument('--weaviate-url', type=str, help='Weaviate URL')
    parser.add_argument('--weaviate-index-name', type=str, default='Wikipedia', help='Weaviate index name')
    parser.add_argument('--faiss-index-path', type=str, help='FAISS Index path')
    parser.add_argument('--dataset-size', type=int, help='Specify a size to reduce the size of the dataset')
    parser.add_argument('--chunk-size', type=int, default=500, help='Chunk size (number of tokens)')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Chunk overlap (number of tokens)')
    parser.add_argument('--min-doc-size', type=int, default=50, help='Minimum document size (number of tokens)')
    parser.add_argument('--num-shards', type=int, default=4, help='Parallelism for embedding calculation')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for embedding calculation')
    parser.add_argument('--num-gpus', type=float, default=0, help='Number of GPUs for embedding calculation. Default is to perform computation on CPU.')

    args = parser.parse_args()
    main(num_shards=args.num_shards, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
         dataset_size=args.dataset_size, faiss_index_path=args.faiss_index_path, min_doc_size=args.min_doc_size,
         batch_size=args.batch_size, weaviate_url=args.weaviate_url, weaviate_index_name=args.weaviate_index_name,
         num_gpus=args.num_gpus)