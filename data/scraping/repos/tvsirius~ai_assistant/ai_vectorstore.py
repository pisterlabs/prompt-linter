import os, json
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import messages_from_dict, messages_to_dict

'''from dotenv import dotenv_values

env_vars = dotenv_values('.env')
OPENAI_API_KEY = env_vars['OPENAI_API_KEY']
'''

embeddings = None
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")
# CHROMA_ID_FULL_JSON = 'history'


if not (os.path.exists(os.path.join(DB_DIR, 'chroma-collections.parquet')) and os.path.exists(
        os.path.join(DB_DIR, 'chroma-embeddings.parquet'))):
    vectorstore = Chroma.from_texts(
        texts=[''],
        embedding=embeddings,
        persist_directory=DB_DIR)
    vectorstore.persist()
else:
    vectorstore = Chroma(
        # collection_name="langchain_store",
        embedding_function=embeddings,
        # client_settings=client_settings,
        persist_directory=DB_DIR,
    )
    vectorstore.persist()


def load_from_vectorstore(id):
    read_history = vectorstore._collection.get(ids=[id], include=["documents"])
    print(read_history)
    if len(read_history['documents']) > 0 and len(read_history['documents'][0]) > 0:
        return json.loads(read_history['documents'][0])


def save_to_vectorstore(id, text):
    vectorstore._collection.upsert(ids=[id], documents=[json.dumps(text)])
    vectorstore.persist()


def load_history(id):
    history_load = load_from_vectorstore(id)
    if history_load:
        return history_load
