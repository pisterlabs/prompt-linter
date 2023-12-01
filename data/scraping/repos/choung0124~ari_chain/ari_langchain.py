import psycopg2
import numpy as np
from tqdm import tqdm
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings

# WHERE hf_BERT_embedding_vector is NULL
# model_name='pritamdeka/S-PubMedBert-MS-MARCO',
# SET hf_BERT_embedding_vector = %s

# WHERE pubmed_bert_vectors is NULL
# model_name='pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb',
# SET pubmed_bert_vectors = %s

def fetch_all_abstracts(conn):
    with conn.cursor() as c:
        c.execute("""
            SELECT pmid, title, doi, abstract
            FROM pubmed_articles
        """)
        return c.fetchall()

with psycopg2.connect(
    dbname='pubmed',
    user="hschoung",
    password="Reeds0124",
    host="localhost",
    port="5432"
) as conn:
    abstracts = fetch_all_abstracts(conn)
    texts = []
    ids = []
    for id, title, abstract, doi in tqdm(abstracts):
        if not abstract:
            continue
        texts.append(abstract)
        ids.append(id)

def embed_and_update_documents_in_batches(model, conn, ids, texts, batch_size=4096):
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = model.embed_documents(batch)

        with conn.cursor() as c:
            for id, embedding in zip(batch_ids, batch_embeddings):
                c.execute("""
                    UPDATE pubmed_articles
                    SET pubmed_bert_vectors = %s
                    WHERE pmid = %s
                """, (embedding, id))
            conn.commit()

hf = HuggingFaceEmbeddings(model_name='pritamdeka/S-PubMedBert-MS-MARCO',
                                model_kwargs= {'device': 'cuda'},
                                encode_kwargs = {'normalize_embeddings': True})

# Update hf_BERT_embedding_vector in the PostgreSQL database with pgvector for every batch
with psycopg2.connect(
    dbname='pubmed',
    user="hschoung",
    password="Reeds0124",
    host="localhost",
    port="5432"
) as conn:
    embed_and_update_documents_in_batches(hf, conn, ids, texts, batch_size=4096)


