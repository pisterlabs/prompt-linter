import os
import json
from dotenv import load_dotenv
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from helpers.logger import get_logger
from helpers.uuid_generator import generate_uuids

load_dotenv()

logger = get_logger()

cloud_config = {"secure_connect_bundle": "creds/secure-connect-vectorstore.zip"}

with open("creds/vectorstore-token.json", "r", encoding="utf-8") as f:
    secrets = json.load(f)

CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")

logger.info("- Starting Vectorstore")
def connection_to_cluster():
    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(
        cloud=cloud_config, 
        auth_provider=auth_provider
        )
    return cluster.connect()

def get_memory_module(session: Cluster) -> ConversationBufferMemory:
    session_id=generate_uuids(1)
    message_history = CassandraChatMessageHistory(
    keyspace=ASTRA_DB_KEYSPACE,
    session=session,
    session_id=session_id,
    ttl_seconds=3600
    )
    #message_history.clear()

    return ConversationBufferMemory(
        memory_key=session_id,
        message_history=message_history,
    )
