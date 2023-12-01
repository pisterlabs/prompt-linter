from typing import Optional, List, Dict, Literal
from datasets import Dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def convert_transcript_chunk(allin_episodes_df) -> List[Dict]:
    allin_episodes_ds = Dataset.from_pandas(allin_episodes_df)
    return [
        dict(
            id=chunk['id'],
            text=chunk['text'],
            metadata=dict(
                episodeTitle=chunk['episodeTitle'],
                start=chunk['start'],
                end_time=chunk['end_time'],
                source='https://www.youtube.com/watch?v=' + ''.join(chunk['id'].split('-')[:-1]),
                channelId= chunk['channelId'],
                chunk=i
            )
        )
        for i, chunk in enumerate(allin_episodes_ds)
    ]

def index_transcript_chunk(allin_episodes_chunks, embedding_model_name: Optional[str] = "text-embedding-ada-002"):
    dimension_map = {
        'text-embedding-ada-002': 1536,
        'sentence-transformers/all-mpnet-base-v2': 768
    }
    if embedding_model_name != 'text-embedding-ada-002':
        embedding_option = 'huggingface'
    else:
        embedding_option = 'openai'
    # initialize embedding engine
    if embedding_option == 'openai':
        embedding_engine = OpenAIEmbeddings()
    else:
       embedding_engine = HuggingFaceEmbeddings(model_name=embedding_model_name)
    texts = [record['text'] for record in allin_episodes_chunks]
    metadatas = [dict(record['metadata']) for record in allin_episodes_chunks]
    # initial setup of FAISS vector store
    faiss_index = FAISS.from_texts(texts, embedding_engine, metadatas)
    return faiss_index