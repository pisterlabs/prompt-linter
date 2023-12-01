import os
from deeplake import VectorStore
from dotenv import load_dotenv
from openai import OpenAI
from embeddings import OpenAIEmbeddings
from synthetic_queries import create_synthetic_queries, load_synthetic_queries
        
client = OpenAI()

load_dotenv()

activeloop_token = os.getenv("ACTIVELOOP_TOKEN")
    

embedding_function = OpenAIEmbeddings(client)

corpus = VectorStore(
    path='hub://activeloop-test/test-deepmemory-adilkhan-autogen',
    token=activeloop_token,
    embedding_function=embedding_function,
    lock_enabled=False,
)

questions, relevance = create_synthetic_queries(corpus, number_of_questions=10, dataset_name="scifact", client=client)

questions, relevance = load_synthetic_queries("scifact_questions_10.txt", "scifact_relevance_10.txt")

corpus.deep_memory.train(
    queries=questions,
    relevance=relevance,
)
