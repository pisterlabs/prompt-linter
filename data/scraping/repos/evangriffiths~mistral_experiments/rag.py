import os
import torch
import transformers

"""
1. Ask an LLM about a recent topic for which it will have no context.
"""
query = (
    "What does Wout Van Aert's 2023-2024 cyclocross race calendar look like? "
    "List the races in chronological order, and include the dates."
)
pipeline = transformers.pipeline(
  model="mistralai/Mistral-7B-Instruct-v0.1",
  device="cuda:0",
  torch_dtype=torch.float16,
)
output = pipeline(query, max_length=500, num_return_sequences=1)[0]['generated_text']
print(output)

"""
2. Create a vector DB, store some articles in this DB, including some useful
   context for answering the query in (1).
"""
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

parent_dir = os.path.dirname(os.path.abspath(__file__))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = []
for f in os.listdir(f"{parent_dir}/context"):
    assert f.endswith(".txt")
    raw_documents = TextLoader(f"{parent_dir}/context/{f}").load()
    documents.extend(text_splitter.split_documents(raw_documents))

db = Chroma.from_documents(documents, HuggingFaceEmbeddings()) # downloads ~500MB model weights
assert db.embeddings.client.device.type == "cuda"

"""
3. Query db for context that matches the query
"""
docs = db.similarity_search(query, k=2)
# Check only chunks from relevant sources are returned from the search
assert all("context/wva-cyclocross-schedule.txt" in d.metadata["source"] for d in docs)
context = " ".join([doc.page_content for doc in docs])

"""
4. Ask the LLM about the same topic as in (1), but this time with added context
"""
enhanced_query = (
    f"Some background info:\n\n{context}\n\n"
    f"[INST]Now answer the following:\n\n{query}\n[/INST]>"
)
output = pipeline(
    enhanced_query,
    max_length=1000,
    num_return_sequences=1
)[0]['generated_text']
print(output)

# TODO: produces correct answers, but then continues with hallucinations until
# `max_length` tokens reached. How to generate 'end of response' token?
