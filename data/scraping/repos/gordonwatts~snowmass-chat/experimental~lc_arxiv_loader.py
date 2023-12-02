# Try to download and "cache" the data loaded from a document on the archive

# from langchain.document_loaders import ArxivLoader
from logging import Logger
import logging
from pathlib import Path

from charset_normalizer import from_path
from .archive_loader import ArxivLoader

document_name = "id:2109.10905"

loader = ArxivLoader(
    document_name, load_all_available_meta=True, doc_content_chars_max=None
)
data = loader.load()

print(len(data))
assert len(data) == 1

good_doc = data[0]

# Next, pickle the good_doc
# json or similar isn't used here b.c. it doesn't support the Document class
# That said, the Document is pydantic 1.0, so it could be there is a better way, I just do
# not know it.
import pickle

with open("good_doc.pickle", "wb") as f:
    pickle.dump(good_doc, f)

# And now read it back in to test that pickle round trip worked.
with open("good_doc.pickle", "rb") as f:
    good_doc_prime = pickle.load(f)
print(len(good_doc_prime.page_content), len(good_doc.page_content))

# Fix up the metadata - can only be a str, int, flaot, or bool.
for k, v in good_doc_prime.metadata.items():
    if isinstance(v, (list, dict)):
        logging.warning(
            f"Metadata {k} is a {type(v)} - {str(v)} - which is not legal! Attempting fix"
        )
        good_doc_prime.metadata[k] = str(v)
    # and if not a legal type, then complain!
    if not isinstance(v, (str, int, float, bool)):
        logging.warning(f"Metadata {k} is a {type(v)} - which is not legal!")

# Next, lets see if we can create the database with this character (e.g. make sure
# nothing that anyone cares about was lost during the pickling).
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents([good_doc_prime])
print(len(all_splits))

# Now, put it in the vector store
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Need api key for this next bit - to access the embeddings.
vector_store_path = Path("vector_store")
print(vector_store_path.absolute())
vector_store_path.mkdir(exist_ok=True)
api_key = Path("openai.key").read_text().strip()
embedding = OpenAIEmbeddings(
    model_name="text-embedding-ada-002", openai_api_key=api_key
)
vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=embedding, persist_directory=str(vector_store_path)
)

# Next, lets ask a question and see what we get back
question = "What does the FASER experiment do?"
docs = vectorstore.similarity_search(question)
len(docs)

# I am really surprised that worked - it got 4 things with FASER mentioned in it.
# Given that almost certainly isn't in the embedding... Can we test the embedding?
emb_vector = embedding.embed_query(question)
print(emb_vector)
print(embedding.embed_query("FASER"))
# Ok - so they are different. Weird!!

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
r = qa_chain({"query": question})
print(r)
