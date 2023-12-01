import time
from pathlib import Path
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

from common import ASSET_PATH, INDEX_NAME


def get_document_paths() -> List[Path]:
  paths = list(path for path in ASSET_PATH.glob('**/*.pdf'))
  paths.sort(key=lambda path: path.name.lower())
  return paths


def load_pages(path:Path) -> List[Document]:
  loader = PyPDFLoader(str(path))
  return loader.load_and_split()


if __name__ == '__main__':
  embeddings = OpenAIEmbeddings()
  faiss = None
  try: faiss = FAISS.load_local(ASSET_PATH, embeddings, INDEX_NAME)
  except: pass

  for document_path in get_document_paths():
    print(str(document_path))

    if faiss is None: faiss = FAISS.from_documents(load_pages(document_path), embeddings)
    else:
      sources = [doc.metadata.get('source') for doc in faiss.docstore._dict.values()]
      if (str(document_path) in sources): continue

      faiss.add_documents(load_pages(document_path))
    time.sleep(0.001)

    faiss.save_local(ASSET_PATH, INDEX_NAME)
