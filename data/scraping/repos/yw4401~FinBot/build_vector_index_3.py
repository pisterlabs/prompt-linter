from contextlib import closing

import datetime

import numpy as np
import pandas as pd
import spacy
import torch
from elasticsearch import Elasticsearch, NotFoundError
from langchain.text_splitter import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
import google.cloud.bigquery as bq
import google.cloud.bigquery.dbapi as bqapi
import re

import config

tqdm.pandas()
if torch.cuda.is_available():
    device = "cuda"
    spacy.prefer_gpu()
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def strip_reuter_intro(text):
    """
    removes the beginning of the reuter intro for reuter articles
    """

    text_chunks = text.split(" - ")
    return " - ".join(text_chunks[1:]).strip()


def create_extractor(ner_recog):
    """
    creates an NER extractor function that will return a list of entities for each given text chunk

    :param ner_recog: the spacy pipeline that extracts entities
    """

    def extractor(chunks):
        """
        extracts the entities in the given chunks

        :param chunks: a list of text chunks
        :returns: a list of list of entities, each list in the list correspond to a text chunk in the input list.
        """

        ner_results = ner_recog.pipe(chunks)
        entities = []
        for doc in ner_results:
            chunk_entities = []
            for ent in doc.ents:
                if ent.label_ in config.ES_ARTICLE_ENTITIES:
                    chunk_entities.append(ent.text)
            entities.append(chunk_entities)
        return entities

    return extractor


def get_doc_chroma_meta(row, chunk_idx):
    """
    .. deprecated:: Using Elastic Search now not Chroma
    """

    published_time = datetime.datetime.fromisoformat(row["published"])
    epoch_time = datetime.datetime(1970, 1, 1, tzinfo=published_time.tzinfo)
    return {
        "title": row["title"],
        # "url": row["url"],
        "published": (published_time - epoch_time).days,
        "doc_id": row["id"],
        "topic": row["topic"],
        "topic_prob": row["probability"],
        "source": row["source"],
        "entity": " ".join(row["entities"][chunk_idx])
    }


def create_splitter():
    """
    creates a LangChain TextSplitter that splits articles into chunks.

    :returns: a text splitter
    """

    tokenizer = AutoTokenizer.from_pretrained(config.ARTICLE_SPLITTER_TOKENIZER,
                                              add_eos_token=False, add_bos_token=False)

    def _huggingface_tokenizer_length(text: str) -> int:
        return len(tokenizer.encode(text))

    splitter = SpacyTextSplitter(length_function=_huggingface_tokenizer_length,
                                 chunk_overlap=config.ARTICLE_SPLITTER_CHUNK_OVERLAP,
                                 chunk_size=config.ARTICLE_SPLITTER_CHUNK_SIZE)
    return splitter


def preprocess_articles(article_df, splitter):
    """
    function that preprocesses the articles before sending it to Elasticsearch. Currently, it will
    strip the intro segment for reuter articles. Then, it will split the texts into chunks, and
    extracts the entities for each chunk.

    :param article_df: the Pandas dataframe containing the articles
    :param splitter: the text splitter
    """

    article_df = article_df.copy()
    clean_newline_regex = re.compile(r"\n+")
    article_df["body"] = article_df.apply(
        lambda row: strip_reuter_intro(row["body"] if row["source"] == "reuters" else row["body"]), axis=1)
    article_df["chunks"] = article_df.body.progress_apply(lambda b: [clean_newline_regex.sub("\n", c).strip() for c in splitter.split_text(b)])
    ner_recog = spacy.load(config.NER_SPACY_MOD, enable=["ner"])
    extractor = create_extractor(ner_recog)
    article_df["entities"] = article_df.chunks.progress_apply(extractor)
    return article_df


def create_es_topic_doc(row, embedding):
    """
    Creates the elastic search documents for a topic

    :param encoder: the embedding model to use
    :param row: a Pandas dataframe row corresponding to a topic
    """

    return {
        "description": row["summary"],
        "description_embedding": embedding.tolist(),
        "metadata": {
            "topic": int(row["topic"]),
            "model": row["model"],
            "recency": row["recency"],
        }
    }


def create_es_topic_idx(client: Elasticsearch, encoder, topic_sum_df):
    """
    Creates or updates a elastic search index for the topics

    :param client: the elastic search client
    :param encoder: the embedding model
    :param topic_sum_df: the dataframe containing the topics and their summaries
    """

    topic_id_format = "{topic_model}-{topic_num}"
    try:
        client.indices.get(index=config.ES_TOPIC_INDEX)
    except NotFoundError as e:
        client.indices.create(index=config.ES_TOPIC_INDEX, mappings=config.ES_TOPIC_MAPPING)
        client.indices.get(index=config.ES_TOPIC_INDEX)

    topic_embeddings = list(encoder.encode(topic_sum_df["summary"], show_progress_bar=True))
    with tqdm(total=topic_sum_df.shape[0]) as progress:
        for i, topic in topic_sum_df.iterrows():
            doc_id = topic_id_format.format(topic_model=topic["model"], topic_num=topic["topic"])
            client.update(index=config.ES_TOPIC_INDEX, id=doc_id,
                          doc=create_es_topic_doc(topic, topic_embeddings[i]), doc_as_upsert=True)
            progress.update(1)
    return topic_embeddings


def create_es_chunk_docs(row, topic_row):
    """
    creates the elastic search document for an article chunl

    :param encoder: the embedding model to use
    :param row: a pandas dataframe row corresponding to a chunk of articles
    """

    published = row["published"].replace(tzinfo=datetime.timezone.utc).strftime('%Y-%m-%d')
    for i, chunk in enumerate(row["chunks"]):
        chunk_id = "{article_id}-{chunk_idx}".format(article_id=row["id"], chunk_idx=i)
        yield chunk_id, {
            "chunk_text": chunk,
            "chunk_text_embedding": [0 for _ in range(1024)],
            "topic_text": topic_row["summary"],
            "topic_text_embedding": topic_row["embeddings"],
            "metadata": {
                "title": row["title"].strip(),
                "url": row["url"].strip(),
                "entities": " ".join(row["entities"][i]),
                "published_at": published,
                "topic": int(row["topic"]),
                "model": row["model"]
            }
        }


def create_es_doc_idx(client: Elasticsearch, encoder, article_df, topic_df):
    """
    creates or update an elastic search index corresponding to the articles

    :param client: the elastic search client
    :param encoder: the embedding model
    :param article_df: a pandas dataframe corresponding to the
    """

    try:
        client.indices.get(index=config.ES_ARTICLE_INDEX)
    except NotFoundError:
        client.indices.create(index=config.ES_ARTICLE_INDEX, mappings=config.ES_ARTICLES_MAPPING)
        client.indices.get(index=config.ES_ARTICLE_INDEX)

    topic_current = topic_df.loc[topic_df.model == article_df.model.iloc[0]]
    mean_embedding = list(np.stack(topic_current.embeddings.tolist()).mean(axis=1))
    article_df = preprocess_articles(article_df, splitter=create_splitter())
    total_chunks = []
    for i, row in article_df.iterrows():
        corresponding_topic = topic_current.loc[topic_df.topic == row["topic"]]
        if corresponding_topic.shape[0] == 0:
            corresponding_topic = {
                "summary": "",
                "embeddings": mean_embedding
            }
        else:
            corresponding_topic = corresponding_topic.iloc[0]
        total_chunks.extend([(id_chunk, doc) for id_chunk, doc in create_es_chunk_docs(row, corresponding_topic)])
    chunk_embeddings = encoder.encode([doc["chunk_text"] for _, doc in total_chunks], show_progress_bar=True)
    for (_, doc), embedding in zip(total_chunks, chunk_embeddings):
        doc["chunk_text_embedding"] = embedding.tolist()

    with tqdm(total=len(total_chunks)) as progress:
        for id_chunk, doc in total_chunks:
            client.update(index=config.ES_ARTICLE_INDEX, id=id_chunk, doc=doc, doc_as_upsert=True)
            progress.update(1)


def get_unindexed_topics(client: bq.Client):
    """
    Gets the latest topics
    """

    query = "SELECT TS.model, TS.topic, TDT.recency, TS.summary FROM " \
            f"(SELECT TAT.model as model, TAT.topic as topic, " \
            f"timestamp_seconds(cast(avg(unix_seconds(CA.published)) as int64)) AS recency " \
            f"FROM Articles.ArticleTopic AS TAT, Articles.CleanedArticles AS CA " \
            f"WHERE TAT.article_id = CA.id AND TAT.topic_prob >= {config.TOPIC_EMBED_TOP_THRESHOLD} " \
            f"GROUP BY TAT.model, TAT.topic) AS TDT, " \
            f"Articles.TopicSummary AS TS, " \
            f"(SELECT TM.id AS id, TM.fit_date FROM Articles.TopicModel as TM " \
            f"ORDER BY TM.fit_date DESC LIMIT 1) AS NM " \
            f"WHERE NM.id = TDT.model AND TDT.model = TS.model AND TDT.topic = TS.topic ORDER BY TS.topic ASC"
    result = []
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query)
            for r in cursor.fetchall():
                result.append(list(r))

    return pd.DataFrame(result, columns=["model", "topic", "recency", "summary"])


def get_articles_by_topics(client: bq.Client, model):
    """
    Gets all articles used by a given revision of the topic model.
    """

    query = ("SELECT CA.id, CA.url, CA.published, CA.source, CA.title, CA.coref, ACT.topic "
             "FROM Articles.CleanedArticles AS CA, "
             "Articles.ArticleTopic ACT "
             "WHERE CA.id = ACT.article_id AND ACT.model = %s")
    result = []
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query, (model,))
            for r in cursor.fetchall():
                result.append(list(r))
    result_df = pd.DataFrame(result, columns=["id", "url", "published", "source", "title", "body", "topic"])
    result_df["model"] = model
    return result_df


def flip_servable(client: bq.Client, model):
    """
    Sets a topic model to be servable
    """
    query = "UPDATE Articles.TopicModel AS TM SET servable=TRUE WHERE TM.id=%s"
    with closing(bqapi.Connection(client=client)) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(query, (model,))
            connection.commit()


if __name__ == "__main__":
    with open(config.ES_KEY_PATH, "r") as fp:
        es_key = fp.read().strip()
    with open(config.ES_CLOUD_ID_PATH, "r") as fp:
        es_id = fp.read().strip()
    print(device)
    elastic_client = Elasticsearch(cloud_id=es_id, api_key=es_key)
    encoder = SentenceTransformer(model_name_or_path=config.TOPIC_FAISS_EMBEDDING, device=device)

    with closing(bq.Client(project=config.GCP_PROJECT)) as client:
        topic_df = get_unindexed_topics(client=client)
        print("Building Topic Indices")
        topic_embeddings = create_es_topic_idx(client=elastic_client, encoder=encoder, topic_sum_df=topic_df)
        topic_df["embeddings"] = topic_embeddings
        for m in topic_df.model.unique():
            article_df = get_articles_by_topics(client=client, model=m)
            print(f"Building Article Indices for {m}")
            create_es_doc_idx(client=elastic_client, encoder=encoder, article_df=article_df, topic_df=topic_df)
            flip_servable(client, m)
