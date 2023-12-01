import sys
sys.path.append("../shared_utils")

import os
import json
import pandas as pd
from tqdm import tqdm
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from sql_engine import SQLEngine
from sklearn.cluster import KMeans

load_dotenv()


class Cluster:
    def __init__(self) -> None:
        self.sql_engine= SQLEngine(
            host= os.getenv("POSTGRESQL_URL"),
            database= "postgres",
            user= "postgres",
            password= os.getenv("POSTGRESQL_PWD"),
            minconn= 1,
            maxconn= 5
        )

    def _tuple_to_dict(self, tuple_data: tuple, keys: List= None) -> dict:
        if not keys: keys= ["id", "title", "url", "embedding", "hn_post_id", "created_at"]
        return dict(zip(keys, tuple_data))
    
    def _get_records(self) -> List[dict]:
        query = \
        """
        SELECT * FROM hn_embeddings
        """
        results = self.sql_engine.execute_select_query(query)
        return [self._tuple_to_dict(result) for result in results]
    
    
    def create_cluster_table(self) -> None:
        query = \
        """
        CREATE TABLE IF NOT EXISTS hn_clusters (
            id bigserial PRIMARY KEY,
            hn_embedding_id bigint REFERENCES hn_embeddings(id),
            cluster_idx int
        )
        """
        self.sql_engine.execute_query(query)
    
    def create_cluster_title_table(self) -> None:
        query = \
        """
        CREATE TABLE IF NOT EXISTS hn_cluster_titles (
            hn_cluster_idx bigserial PRIMARY KEY,
            title varchar(255)
        )
        """
        self.sql_engine.execute_query(query)

    def insert_to_cluster_table(self, data: List[dict]) -> None:
        query = \
        """
        INSERT INTO hn_clusters (hn_embedding_id, cluster_idx)
        VALUES (%s, %s)
        """
        for record in data:
            self.sql_engine.execute_insert_query(query, (record["id"], record["cluster_idx"]))

    def insert_to_cluster_title_table(self, data: List) -> None:
        query = \
        """
        INSERT INTO hn_cluster_titles(hn_cluster_idx, title)
        VALUES (%s, %s)
        ON CONFLICT (hn_cluster_idx)
        DO 
            UPDATE SET title = EXCLUDED.title
        """
        self.sql_engine.execute_insert_query(query, (data[0], data[1]))

    def get_clusters(self) -> List[dict]:
        query = \
        """
        SELECT * FROM hn_clusters
        """
        return self.sql_engine.execute_select_query(query)

    def get_clustered_data(self) -> List[dict]:
        query = \
        """
        SELECT * FROM hn_embeddings
        INNER JOIN hn_clusters ON hn_embeddings.id = hn_clusters.hn_embedding_id
        """
        results = self.sql_engine.execute_select_query(query)
        return [self._tuple_to_dict(result) for result in results]

    def get_unqiue_cluster_idx(self) -> List[int]:
        query = \
        """
        SELECT DISTINCT cluster_idx FROM hn_clusters
        """
        results = self.sql_engine.execute_select_query(query)
        return [result[0] for result in results]

    def get_records_by_cluster_idx(self, cluster_idx: int, limit: int= 5) -> List[dict]:
        query = \
        """
        SELECT * FROM hn_embeddings
        INNER JOIN hn_clusters ON hn_embeddings.id = hn_clusters.hn_embedding_id
        WHERE hn_clusters.cluster_idx = %s
        LIMIT %s
        """
        results = self.sql_engine.execute_select_query(query, (cluster_idx, limit))
        return [self._tuple_to_dict(result) for result in results]

    def execute_cluster(self) -> None:
        clusters = self.get_clusters()
        if len(clusters)>0: return

        records = self._get_records()
        embeddings = [json.loads(record["embedding"]) for record in records]
        kmeans = KMeans(n_clusters= 10, random_state= 0).fit(embeddings)
        cluster_labels = kmeans.labels_
        for i, record in enumerate(records):
            record["cluster_idx"] = int(cluster_labels[i])
        
        self.insert_to_cluster_table(records)
        return records
    
    def pre_processing(self) -> None:
        self.create_cluster_table()
        self.create_cluster_title_table()
        self.execute_cluster()

    def _call_api(self) -> str:
        raise NotImplementedError("_call_api method is not implemented")

    def process(self) -> None:
        raise NotImplementedError("process method is not implemented")

