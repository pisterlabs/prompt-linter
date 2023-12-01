import lancedb
import re
import pickle
import requests
import zipfile
import pandas as pd
from pathlib import Path

from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import LanceDB

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_document_title(document):
    m = str(document.metadata["source"])
    title = re.findall("pandas.documentation(.*).html", m)
    if title[0] is not None:
        return(title[0])
    return ''

def embed_fun(text):
    return [model.encode(x) for x in text]


if __name__ == "__main__":
    query = ""

    docs_path = Path("docs.pkl")
    docs = []

    if not docs_path.exists():
        pandas_docs = requests.get("https://eto-public.s3.us-west-2.amazonaws.com/datasets/pandas_docs/pandas.documentation.zip")
        with open('./tmp/pandas.documentation.zip', 'wb') as f:
            f.write(pandas_docs.content)

        file = zipfile.ZipFile("./tmp/pandas.documentation.zip")
        file.extractall(path="./tmp/pandas_docs")

        for p in Path("./tmp/pandas_docs/pandas.documentation").rglob("*.html"):
            print(p)
            if p.is_dir():
                continue
            loader = BSHTMLLoader(p, open_encoding="utf8")
            raw_document = loader.load()

            m = {}
            m["title"] = get_document_title(raw_document[0])
            m["version"] = "2.0rc0"
            raw_document[0].metadata = raw_document[0].metadata | m
            raw_document[0].metadata["source"] = str(raw_document[0].metadata["source"])
            docs = docs + raw_document

        with docs_path.open("wb") as fh:
            pickle.dump(docs, fh)
    else:
        with docs_path.open("rb") as fh:
            docs = pickle.load(fh)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(docs)

    db = lancedb.connect('./tmp/lancedb')

    data = [doc.page_content for doc in documents]
    data = pd.DataFrame(data, columns=["text"])
    print(data)

    table = db.create_table("pandas_docs", data, embed_fun)

    table = db.open_table("pandas_docs")
    print(table.to_pandas())
