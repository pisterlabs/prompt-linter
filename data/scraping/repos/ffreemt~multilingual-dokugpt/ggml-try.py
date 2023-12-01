"""Adopted from https://github.com/imartinez/privateGPT/blob/main/privateGPT.py

https://raw.githubusercontent.com/imartinez/privateGPT/main/requirements.txt

from pathlib import Path
Path("models").mkdir(exit_ok=True)
!time wget -c https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin -O models/ggml-gpt4all-j-v1.3-groovy.bin"""
from dotenv import load_dotenv, dotenv_values
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time

from types import SimpleNamespace
from chromadb.config import Settings

# embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
# persist_directory = os.environ.get('PERSIST_DIRECTORY')

# load_dotenv()
# model_type = os.environ.get('MODEL_TYPE')
# model_path = os.environ.get('MODEL_PATH')
# model_n_ctx = os.environ.get('MODEL_N_CTX')
# model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
# target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

settings = dict([('PERSIST_DIRECTORY', 'db1'),
             ('MODEL_TYPE', 'GPT4All'),
             ('MODEL_PATH', 'models/ggml-gpt4all-j-v1.3-groovy.bin'),
             ('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2'),
             ('MODEL_N_CTX', '1000'),
             ('MODEL_N_BATCH', '8'),
             ('TARGET_SOURCE_CHUNKS', '4')])

# models/ggml-gpt4all-j-v1.3-groovy.bin  ~5G

persist_directory = settings.get('PERSIST_DIRECTORY')

model_type = settings.get('MODEL_TYPE')
model_path = settings.get('MODEL_PATH')
embeddings_model_name = settings.get("EMBEDDINGS_MODEL_NAME")
# embeddings_model_name = 'all-MiniLM-L6-v2'
# embeddings_model_name = 'paraphrase-multilingual-mpnet-base-v2'

model_n_ctx = settings.get('MODEL_N_CTX')
model_n_batch = int(settings.get('MODEL_N_BATCH',8))
target_source_chunks = int(settings.get('TARGET_SOURCE_CHUNKS',4))

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
)

args = SimpleNamespace(hide_source=False, mute_stream=False)

# load chroma database from db1
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
# activate/deactivate the streaming StdOut callback for LLMs
callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
# Prepare the LLM
match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    case "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    case _default:
        # raise exception if model_type is not supported
        raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

# need about 5G RAM

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

# Get the answer from the chain

query = "共产党是什么"

start = time.time()
res = qa(query)
answer, docs = res['result'], [] if args.hide_source else res['source_documents']
end = time.time()

# Print the result
print("\n\n> Question:")
print(query)
print(f"\n> Answer (took {round(end - start, 2)} s.):")
print(answer)
