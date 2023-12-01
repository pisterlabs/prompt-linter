import sys
import os
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

import docgrab

INTRO_ASCII_ART = """ ,___,   ,___,   ,___,                                                 ,___,   ,___,   ,___,
 [OvO]   [OvO]   [OvO]                                                 [OvO]   [OvO]   [OvO]
 /)__)   /)__)   /)__)    WELCOME TO DOCDOCGO PREREQUISITE VERIFIER    /)__)   /)__)   /)__)
--"--"----"--"----"--"--------------------------------------------------"--"----"--"----"--"--"""

VERBOSE = True


def create_test_vectorstore(docs, save_dir):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs = text_splitter.split_documents(docs)

        # As of Aug 8, 2023, max chunk size for Azure API is 16
        embeddings = OpenAIEmbeddings(
            deployment="text-embedding-ada-002", chunk_size=16
        )

        # Only put the first two documents in the vectorstore for testing
        vectorstore = Chroma.from_documents(
            docs[:2], embeddings, persist_directory=save_dir
        )
        return vectorstore
    except Exception as e:
        # Print authentication errors etc.
        print(e)
        sys.exit()


print(INTRO_ASCII_ART + "\n\n")
load_dotenv()

# Download Confluence pages and save to jsonl
docs = docgrab.load_confluence()
num_docs = len(docs)
print(f"1. Loaded {num_docs} documents from Confluence.")

jsonl_path = os.path.join(
    os.getenv("SAVE_CONFLUENCE_DIR"),
    f"confluence-space-{os.getenv('CONFLUENCE_SPACE')}.jsonl",
)
docgrab.save_docs_to_jsonl(
    docs, jsonl_path, mode="w"
)  # for testing, overwrite previous docs
print(f"2. Saved documents to {jsonl_path}.")

# Create vectorstore from documents
loader = docgrab.JSONLDocumentLoader(jsonl_path)
docs = loader.load()
if num_docs != len(docs):
    raise ValueError("The number of loaded documents does not match the number saved.")

while True:
    ans = input(
        ">> We are about to test the ability to send requests to the OpenAI API."
        " This will incur a tiny cost (less than $0.01). Continue? (y/n) "
    )
    if ans == "y":
        break
    if ans == "n":
        sys.exit()

vectorstore = create_test_vectorstore(docs, save_dir=os.getenv("SAVE_VECTORDB_DIR"))
print(
    f"3. Created a small test vectorstore and tested ability to access the OpenAI API."
)
