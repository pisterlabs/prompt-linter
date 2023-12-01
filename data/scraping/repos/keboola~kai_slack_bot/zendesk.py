import os
import pandas as pd
import json
import pickle
import pinecone
import time
from dotenv import load_dotenv

from llama_index import (
    GPTVectorStoreIndex,
    ServiceContext,
    Document,
    set_global_service_context,
)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import OpenAI
from llama_index.llm_predictor import LLMPredictor
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import (
    MetadataExtractor,
    # TitleExtractor,
    QuestionsAnsweredExtractor,
    # SummaryExtractor,
    KeywordExtractor,
)
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from tqdm import tqdm

from kbcstorage.client import Client

# Load environment variables from .env file
dotenv_path = '/Users/jordanburger/Keboola/Keboola AI (Kai)/Knowledge Base Chatbot/slack_flask_app/.env'
load_dotenv(dotenv_path)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
KBC_STORAGE_API_TOKEN = os.environ.get('KBC_STORAGE_API_TOKEN')
BUCKET_ID = os.environ.get('BUCKET_ID')

# Folder to save downloaded files
SAVE_FOLDER = '/Users/jordanburger/Keboola/Keboola AI (Kai)/Knowledge Base Chatbot/slack_flask_app/database/datadir/zendesk'
LIMIT = 100


class ZendeskDataExtractor:
    def __init__(self):
        self.client = None

    def authenticate(self):
        self.client = Client('https://connection.keboola.com',
                             os.environ['KBC_STORAGE_API_TOKEN'])

    def get_zendesk_data(self):
        if self.client is None:
            self.authenticate()

        tables = self.client.buckets.list_tables(bucket_id=os.environ['BUCKET_ID'])
        tables_df = pd.DataFrame(tables)

        for table in tqdm(tables_df.itertuples(), desc="Downloading tables from SAPI."):
            self.client.tables.export_to_file(
                table_id=table[2], path_name=f'{SAVE_FOLDER}/raw/')

    def create_zd_ticket_files(self):
        tickets = pd.read_csv(f'{SAVE_FOLDER}/raw/tickets')
        comments = pd.read_csv(f'{SAVE_FOLDER}/raw/tickets_comments')

    # Create a dictionary with ticket IDs as keys and lists of comments as values
    comments_dict = {}
    for _, comment in comments.iterrows():
        ticket_id = comment['tickets_pk']
        comment_body = comment['body']
        comment_id = comment['id']
        comment_created_at = comment['created_at']
        comment_author_pk = comment['author_pk']

        comment_dict = {
            'Comment ID': comment_id,
            'Comment Body': comment_body,
            'Comment Created At': comment_created_at,
            'Comment Author PK': comment_author_pk
        }

        if ticket_id not in comments_dict:
            comments_dict[ticket_id] = []
        comments_dict[ticket_id].append(comment_dict)

    ticket_data = []

    for _, ticket in tickets.iterrows():
        ticket_id = ticket['id']
        ticket_subject = ticket['subject']
        ticket_status = ticket['status']
        ticket_type = ticket['type']
        created_at = ticket['created_at']
        updated_at = ticket['updated_at']

        ticket_dict = {
            'Ticket ID': ticket_id,
            'Ticket Subject': ticket_subject,
            'Ticket Status': ticket_status,
            'Ticket Type': ticket_type,
            'Created At': created_at,
            'Updated At': updated_at,
            # Default to an empty list if no comments are found
            'Comments': comments_dict.get(ticket_id, [])
        }

        ticket_data.append(ticket_dict)

    for ticket in ticket_data:
        ticket_id = ticket.get("Ticket ID")
        with open(f'{SAVE_FOLDER}/tickets/{ticket_id}.json', 'w') as f:
            json.dump(ticket, f)


class DocumentIndexCreator:
    def __init__(self, save_folder, index_filename, batch_size=100):
        self.save_folder = save_folder
        self.index_filename = index_filename
        self.batch_size = batch_size
        self.doc_titles = []
        self.doc_paths = []
        self.doc_ids = []
        self.doc_embeddings = []

        self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=256)
        self.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
        self.llm_predictor = LLMPredictor(llm=self.llm)
        self.text_splitter = TokenTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)
        self.metadata_extractor = MetadataExtractor(
            extractors=[
                # TitleExtractor(nodes=2),
                QuestionsAnsweredExtractor(
                    questions=3, llm_predictor=self.llm_predictor
                ),
                # SummaryExtractor(summaries=["prev", "self"]),
                KeywordExtractor(keywords=5, llm_predictor=self.llm_predictor),
            ]
        )
        self.node_parser = SimpleNodeParser(
            text_splitter=self.text_splitter, metadata_extractor=self.metadata_extractor
        )

        self.load_index()

    def sanitize_filename(self, filename):
        return re.sub(r"[/\\]", "_", filename)

    def read_file_as_string(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def get_file_metadata(self, file_path):
        metadata_path = file_path.replace(".txt", ".json")
        md = self.read_file_as_string(metadata_path)
        md = json.loads(md)
        if md:
            return md
        return {}

    def load_documents(self):
        for dirpath, dirnames, filenames in os.walk(self.save_folder):
            for filename in filenames:
                if filename.endswith(".txt"):
                    subdir_name = os.path.basename(dirpath)
                    file_name = os.path.splitext(filename)[0]

                    self.doc_titles.append(subdir_name + " - " + file_name)
                    self.doc_paths.append(os.path.join(dirpath, filename))

    def index_documents(self):
        nodes = []
        for title, path in zip(self.doc_titles, self.doc_paths):
            if path.endswith(".txt"):
                text = self.read_file_as_string(path)
                extra_info = self.get_file_metadata(path)

                nodes.append(Document(text=text, doc_id=title, extra_info=extra_info))
                print("Document added: " + title)

                if len(nodes) >= self.batch_size:
                    self.process_batch(nodes)
                    nodes = []

        if nodes:
            self.process_batch(nodes)

    def process_batch(self, nodes):
        service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=self.embed_model, node_parser=self.node_parser
        )
        set_global_service_context(service_context)

        start = time.time()
        print(time.time())

        parsed_nodes = self.node_parser.get_nodes_from_documents(nodes, show_progress=True)

        print(time.time() - start)
        print("Nodes added: " + str(len(parsed_nodes)))

        self.update_index(parsed_nodes)

    def save_index(self):
        with open(self.index_filename, "wb") as file:
            index_data = {
                "doc_ids": self.doc_ids,
                "doc_embeddings": self.doc_embeddings,
            }
            pickle.dump(index_data, file)

    def load_index(self):
        if os.path.exists(self.index_filename):
            with open(self.index_filename, "rb") as file:
                index_data = pickle.load(file)
                self.doc_ids = index_data.get("doc_ids", [])
                self.doc_embeddings = index_data.get("doc_embeddings", [])

    def update_index(self, nodes):
        for node in nodes:
            if node.ref_doc_id not in self.doc_ids:
                self.doc_ids.append(node.ref_doc_id)
                self.doc_embeddings.append(node.embedding)
            else:
                index = self.doc_ids.index(node.ref_doc_id)
                self.doc_embeddings[index] = node.embedding

        self.save_index()


def create_and_load_index(index_name, nodes):
    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )

    pinecone_index = pinecone.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002", embed_batch_size=100
    )
    service_context = ServiceContext.from_defaults(embed_model=embed_model)

    GPTVectorStoreIndex(
        nodes,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )

if __name__ == "__main__":
    # Load environment variables from .env file
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)

    # Constants
    SAVE_FOLDER = "downloaded_data"
    INDEX_NAME = "kaidev"
    INDEX_FILENAME = "index_data.pkl"
    BATCH_SIZE = 50

    downloader = ZendeskDataExtractor()
    downloader.get_zendesk_data()

    indexer = DocumentIndexCreator(
        save_folder=SAVE_FOLDER, index_filename=INDEX_FILENAME, batch_size=BATCH_SIZE
    )
    indexer.load_documents()
    indexer.index_documents()

    create_and_load_index(index_name=INDEX_NAME, nodes=indexer.doc_ids, )  # Only push document IDs to Pinecone

