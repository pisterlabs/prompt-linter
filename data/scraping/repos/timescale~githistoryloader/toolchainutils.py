from abc import abstractmethod
import datetime
import openai
from typing import List, Tuple
from llama_index.vector_stores import TimescaleVectorStore
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import TimescaleVector
import pandas as pd
from pandas import DataFrame
from timescale_vector import client
from llama_index.schema import TextNode
from llama_index.embeddings import OpenAIEmbedding
from git import Repo

MAX_STR_LENGTH = 2048
EMBEDDING_DIMENSIONS = 1536
class ToolChain:
    def __init__(self, repo_dir, table_name, toolchain) -> None:
        self._repo_dir= repo_dir
        self._table_name = table_name
        self._tool_chain = toolchain
        self._time_delta = timedelta(days=7)
        openai.api_key  = os.environ['OPENAI_API_KEY']
        self._records = []

    def get_table_name(self)->str:
        return self._table_name

    def get_tool_chain(self)->str:
        return self._tool_chain
    
    @abstractmethod
    def create_tables(self):
        pass

    @abstractmethod
    def process_frame(self, df):
        pass

    @abstractmethod
    def insert_rows(self, rows):
        pass

    @abstractmethod
    def create_index(self):
        pass

    # Helper function: get embeddings for a text
    def get_embeddings(self, text):
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input = text.replace("\n"," ")
        )
        embedding = response['data'][0]['embedding']
        return embedding

    # helper function to take in a date string in the past and return a uuid v1
    def create_uuid(self, date_string: str):
        datetime_obj = datetime.fromisoformat(date_string)
        uuid = client.uuid_from_time(datetime_obj)
        return str(uuid)

    def process_commit_range(self, commit_count, skip_count):
        repo = Repo(self._repo_dir)
        # Create lists to store data
        commit_hashes = []
        authors = []
        dates = []
        subjects = []
        bodies = []
        # Iterate through commits and collect data
        for commit in repo.iter_commits(max_count=commit_count, skip=skip_count):
            commit_hash = commit.hexsha
            author = commit.author.name
            date = commit.committed_datetime.isoformat()
            message_lines = commit.message.splitlines()
            subject = message_lines[0]
            body = "\n".join(message_lines[1:]) if len(message_lines) > 1 else ""

            commit_hashes.append(commit_hash)
            authors.append(author)
            dates.append(date)
            subjects.append(subject)
            bodies.append(body)

        # Create a DataFrame from the collected data
        data = {
            "Commit Hash": commit_hashes,
            "Author": authors,
            "Date": dates,
            "Subject": subjects,
            "Body": bodies
        }
        df = pd.DataFrame(data)
        df.dropna(inplace=True)
        df = df.astype(str)
        df = df.applymap(lambda x: x.strip('"'))
        #print(df.iloc[[0, -1]])
        return(df)

class LangChain(ToolChain):
    def __init__(self, file_name, table_name) -> None:
        super().__init__(file_name, table_name, "langchain")
        self._ts_vector_store = TimescaleVector(
            service_url=os.environ["TIMESCALE_SERVICE_URL"],
            embedding=EMBEDDING_DIMENSIONS,
            collection_name=self._table_name,
            time_partition_interval=self._time_delta
        )
        self._records = []


    def create_tables(self):
        self._ts_vector_store.sync_client.drop_table()
        self._ts_vector_store.sync_client.create_tables()

    def process_row(self, row) -> any:
        max_retries = 2  # Number of times to retry
        text = row['Author'] + " " + row['Date'] + " " + row['Commit Hash'] + " " + row['Subject'] + " " + row['Body']
        record = None
        for _ in range(max_retries):
            try:
                embedding = self.get_embeddings(text)    
                uuid = self.create_uuid(row['Date'])
                metadata = {
                    "author": row['Author'],
                    "date": row['Date'],
                    "commit": row['Commit Hash'],
                }
                record = (uuid, metadata, text, embedding)
                break
            except Exception as e:
                print(f"An exception occurred: {e} Retrying")
                if len(text) > MAX_STR_LENGTH:
                    text = text[:MAX_STR_LENGTH]
        else:
            print(f"Unable to add the record {text}")
        return record

    def process(self, commit_count, skip_count):
        df = self.process_commit_range(commit_count, skip_count)
        self._records = []
        for _, row in df.iterrows():
            record = self.process_row(row)
            if record:
                self._records.append(record)

    def save(self):
        if (len(self._records)) < 1:
            return
        print(f"Inserting {len(self._records)} records")
        self._ts_vector_store.sync_client.upsert(self._records)

    def create_index(self):
        print("Creating Index")
        self._ts_vector_store.create_index()

class LlamaIndex(ToolChain):
    def __init__(self, file_name, table_name) -> None:
        super().__init__(file_name, table_name, "llamaindex")
        self._nodes = []
        self._ts_vector_store = TimescaleVectorStore.from_params(
            service_url=os.environ["TIMESCALE_SERVICE_URL"],
            table_name=self._table_name,
            time_partition_interval=self._time_delta,
        )
        self._nodes=[]

    def create_tables(self):
        self._ts_vector_store._sync_client.drop_table()
        self._ts_vector_store._sync_client.create_tables()

    # Create a Node object from a single row of data
    def create_node(self, row):
        record = row.to_dict()
        record_content = (
            str(record["Date"])
            + " "
            + record['Author']
            + " "
            + str(record["Subject"])
            + " "
            + str(record["Body"])
        )
        # Can change to TextNode as needed
        node = TextNode(
            id_=self.create_uuid(record["Date"]),
            text=record_content,
            metadata={
                "commit_hash": record["Commit Hash"],
                "author": record['Author'],
                "date": record["Date"],
            },
        )
        return node    
    
    def process(self, commit_count, skip_count):
        df = self.process_commit_range(commit_count, skip_count)
        self._nodes = [self.create_node(row) for _, row in df.iterrows()]
        embedding_model = OpenAIEmbedding()
        embedding_model.api_key = os.environ['OPENAI_API_KEY']
        for node in self._nodes:
            node_embedding = embedding_model.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding

    def save(self):
        if (len(self._nodes)) < 1:
            return
        print(f"Inserting {len(self._nodes)} records")
        ts_vector_store = TimescaleVectorStore.from_params(
            service_url=os.environ["TIMESCALE_SERVICE_URL"],
            table_name=self._table_name,
            time_partition_interval=self._time_delta,
        )
        ts_vector_store.add(self._nodes)

    def create_index(self):
        print("Creating Index")
        self._ts_vector_store.create_index()
