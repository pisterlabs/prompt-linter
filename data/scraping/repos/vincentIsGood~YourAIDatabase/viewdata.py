import argparse
from datetime import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import configs.common as config
from configs.common import model_config
from lib.utils.InteractiveConsole import InteractiveConsole, SimpleCommandHandler
from lib.docloader import parseDateTime, fromTimestamp

cmdParser = argparse.ArgumentParser()
cmdParser.add_argument("-id", "--docid", default=None, type=str, help="Print content of a document")
cmdParser.add_argument("-it", "--interactive", action="store_true", default=False, help="Interactive mode")
cmdParsed = cmdParser.parse_args()

print("[+] Preparing Chroma DB")
embedding_func = SentenceTransformerEmbeddings(model_name=model_config.SENTENCE_EMBEDDING_MODEL, cache_folder=config.CACHE_DIR)
chromadb = Chroma(persist_directory=config.PERSIST_DIRECTORY, embedding_function=embedding_func)

print("[+] Chroma index:")
print(chromadb._collection, "\n")

print("[+] Chroma # of collections: ", chromadb._collection.count())

def viewAllDocs():
    collection = chromadb._collection.get(include=["metadatas"])
    printResults(collection["ids"], collection["metadatas"])

def viewDocsSinceDate(args: 'list[str]'):
    # https://docs.trychroma.com/usage-guide?lang=py#using-where-filters
    date = args[0]
    time = args[1] if len(args) > 1 else "0:0:0"
    print("[+] Showing content for docs entered db since %s %s" % (date, time))
    collection = chromadb._collection.get(where={
        "time": {
            "$gt": parseDateTime(date + " " + time)
        }
    }, include=["metadatas", "documents"])

    printResults(collection["ids"], collection["metadatas"])

def viewSpecificDoc(id: str):
    print("[+] Showing content for doc with id: %s" % id)
    collection = chromadb._collection.get(ids=[id], include=["metadatas", "documents"])
    print(collection["metadatas"])
    print(collection["documents"])

def printResults(ids, metadatas):
    print("[+] Documents (ID -> Metadata)")
    for i in range(len(ids)):
        metadatas[i]["time"] = fromTimestamp(metadatas[i]["time"])
        print(f"[*] '{ids[i]}': {metadatas[i]}")

if cmdParsed.interactive:
    print("[+] Entering interactive mode")
    console = InteractiveConsole()
    console.addHandler(SimpleCommandHandler(lambda args: viewSpecificDoc(args[0]), "docid", "view document with content with its ID"))
    console.addHandler(SimpleCommandHandler(lambda args: viewAllDocs(), "docs", "view all documents with its content"))
    console.addHandler(SimpleCommandHandler(lambda args: viewDocsSinceDate(args), "time", "view all documents since a date in format '%d/%m/%Y %H:%M:%S'"))
    console.takeover()
elif cmdParsed.docid:
    viewSpecificDoc(cmdParsed.docid)
else: 
    viewAllDocs()