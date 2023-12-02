import tiktoken
from langchain.schema import Document
from langchain.vectorstores import Pinecone

docs = []

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

for chunk in text.split("\n\n"):
    if chunk:
        docs.append(Document(page_content=chunk))

for doc in docs:
    print(doc.page_content)
    print("\n======\n")