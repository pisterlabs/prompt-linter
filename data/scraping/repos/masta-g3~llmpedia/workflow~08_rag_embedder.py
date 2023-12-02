from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.docstore.document import Document
from langchain.embeddings import CohereEmbeddings

from sqlalchemy.exc import IntegrityError, OperationalError
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import os, sys
import time
import json

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.environ.get("PROJECT_PATH"))
chunk_path = os.path.join(os.environ.get("PROJECT_PATH"), "data", "arxiv_chunks")

import utils.paper_utils as pu
import utils.db as db

MAX_RETRIES = 3
RETRY_DELAY = 2

COLLECTION_NAMES = [
    "arxiv_vectors",
    "arxiv_vectors_cv3",
]

MODEL_NAME_MAP = {
    "arxiv_vectors": "thenlper/gte-large",
    "arxiv_vectors_cv3": "embed-english-v3.0",
}


def main():
    """Create embeddings for all arxiv chunks and upload them to DB."""
    CONNECTION_STRING = (
        f"postgresql+psycopg2://{pu.db_params['user']}:{pu.db_params['password']}"
        f"@{pu.db_params['host']}:{pu.db_params['port']}/{pu.db_params['dbname']}"
    )
    for COLLECTION_NAME in COLLECTION_NAMES:
        print(f"Processing {COLLECTION_NAME}...")
        model_name = MODEL_NAME_MAP[COLLECTION_NAME]

        if "embed-english" in model_name:
            embeddings = CohereEmbeddings(
                cohere_api_key=os.getenv("COHERE_API_KEY"), model=model_name
            )
        else:
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

        store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            embedding_function=embeddings,
        )

        arxiv_codes = db.get_arxiv_id_embeddings(pu.db_params, COLLECTION_NAME)
        local_codes = os.listdir(chunk_path)
        local_codes = [code.replace(".json", "") for code in local_codes]
        processing_codes = list(set(local_codes) - set(arxiv_codes))

        for arxiv_code in tqdm(processing_codes):
            chunks_fname = os.path.join(chunk_path, f"{arxiv_code}.json")
            chunks_json = json.load(open(chunks_fname, "r"))
            chunks_df = pd.DataFrame(chunks_json)
            add_count = 0
            metadata = None
            for idx, row in chunks_df.iterrows():
                chunk_text = row["text"]
                metadata = row.drop("text").to_dict()
                metadata["model"] = model_name
                for attempt in range(MAX_RETRIES):
                    try:
                        store.add_documents(
                            [Document(page_content=chunk_text, metadata=metadata)]
                        )
                        add_count += 1
                        break
                    except IntegrityError as e:
                        continue
                    except OperationalError as e:
                        print(
                            f"Encountered error on paper {metadata['arxiv_code']}: {e}"
                        )
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)
                    continue
            # if metadata:
            #     print(f"Added {add_count} vectors for {metadata['arxiv_code']}.")

        print("Process complete.")


if __name__ == "__main__":
    main()
