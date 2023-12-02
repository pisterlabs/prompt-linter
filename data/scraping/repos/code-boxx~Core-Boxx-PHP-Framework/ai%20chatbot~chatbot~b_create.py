# (A) LOAD SETTINGS & MODULES
import a_settings as set
import os, glob
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
  CSVLoader,
  PyMuPDFLoader,
  TextLoader,
  UnstructuredEPubLoader,
  UnstructuredHTMLLoader,
  UnstructuredMarkdownLoader,
  UnstructuredODTLoader,
  UnstructuredPowerPointLoader,
  UnstructuredWordDocumentLoader
)

# (B) HELPER - RECURSIVE DELETE FOLDER
def rmdir(folder):
  folder = Path(folder)
  for sub in folder.iterdir():
    if sub.is_dir():
      rmdir(sub)
    else:
      sub.unlink()
  folder.rmdir()

# (C) DOCUMENTS & CHECKS
# (C1) REMOVE EXISTING DATABASE
if os.path.exists(set.path_db):
  print("Removing existing - " + set.path_db)
  rmdir(set.path_db)

# (C2) MAP FILE TYPES TO RESPECTIVE LOADER
mapload = {
  ".csv" : CSVLoader,
  ".pdf" : PyMuPDFLoader,
  ".txt" : TextLoader,
  ".epub" : UnstructuredEPubLoader,
  ".html" : UnstructuredHTMLLoader,
  ".md" : UnstructuredMarkdownLoader,
  ".odt" : UnstructuredODTLoader,
  ".doc" : UnstructuredWordDocumentLoader, ".docx" : UnstructuredWordDocumentLoader,
  ".ppt": UnstructuredPowerPointLoader, ".pptx": UnstructuredPowerPointLoader
}

# (C3) GET DOCUMENTS TO IMPORT
all = []
for ext in mapload:
  all.extend(glob.glob(os.path.join(set.path_docs, "*" + ext), recursive=True))
if (len(all) == 0):
  print("No documents to import in " + set.path_docs)
  exit()

# (D) IMPORT PROCESS
print("Creating database")

# (D1) CREATE EMPTY-ISH DATABASE
db = Chroma.from_texts(
  texts = [""],
  embedding = HuggingFaceInstructEmbeddings(**set.embed_args),
  persist_directory = set.path_db
)
db.persist()

# (D2) ADD DOCUMENTS
splitter = RecursiveCharacterTextSplitter(**set.db_split)
for doc in all:
  print("Adding - " + doc)
  name, ext = os.path.splitext(doc)
  db.add_documents(splitter.split_documents(mapload[ext](doc).load()))
db.persist()
db = None
print("Done")