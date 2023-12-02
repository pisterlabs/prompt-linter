"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
import pickle
import os

os.environ["OPENAI_API_KEY"] = "<your-api-key>"

# Here we load in the data in the format that Notion exports it in.
md_dir = "/home/azureuser/OpenAI-Hack-Group-5-Humanitarian/Python/app/"
ps = list(Path(md_dir).glob("train.txt"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p.name)

# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
#text_splitter = CharacterTextSplitter(chunk_size=1500, separator=". ", chunk_overlap=0)
text_splitter = CharacterTextSplitter(chunk_size=200, separator=". ")

docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

print(docs)
print(metadatas)
# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
# store = FAISS.from_texts(docs, HuggingFaceInstructEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
