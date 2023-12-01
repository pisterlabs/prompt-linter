import json
import chromadb
from datasets import load_dataset
from langchain.embeddings import HuggingFaceBgeEmbeddings


model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedder = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

ds = load_dataset("hyperdemocracy/uscb.s1024.o256.bge-small-en", split="train")
chroma_client = chromadb.PersistentClient(path="chroma.db")
collection = chroma_client.create_collection(
    name='uscb',
    embedding_function=embedder.embed_documents,
)
batch_size = 128

for batch in ds.iter(batch_size=batch_size):

    metadatas = []
    for metadata in batch["metadata"]:
        dd = metadata.copy()
        dd["sponsors"] = json.dumps(metadata["sponsors"])
        dd["cosponsors"] = json.dumps(metadata["cosponsors"])
        dd["subjects"] = json.dumps(metadata["subjects"])
        dd["policy_area"] = dd["policy_area"] or ""
        dd["summary_text"] = dd["summary_text"] or ""

        for k,v in dd.items():
            if v is None:
                print(dd)

        metadatas.append(dd)

    collection.add(
        documents=batch["text"],
        metadatas=metadatas,
        embeddings=batch["vec"],
        ids=batch["id"],
    )
