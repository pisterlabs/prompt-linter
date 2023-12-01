#!pip install InstructorEmbedding

import environment

import os
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from huggingface_hub import snapshot_download

# download the vectorstore for the book you want
repo_id="calmgoose/book-embeddings"
BOOK="1984"
cache_dir=f"{BOOK}_cache"
vectorstore = snapshot_download(repo_id=repo_id,
                                repo_type="dataset",
                                revision="main",
                                allow_patterns=f"books/{BOOK}/*", # to download only the one book
                                cache_dir=cache_dir,
                                )

# get path to the `vectorstore` folder that you just downloaded
# we'll look inside the `cache_dir` for the folder we want
target_dir = BOOK

# Walk through the directory tree recursively
for root, dirs, files in os.walk(cache_dir):
    # Check if the target directory is in the list of directories
    if target_dir in dirs:
        # Get the full path of the target directory
        target_path = os.path.join(root, target_dir)

# load embeddings
# this is what was used to create embeddings for the book
embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the book passage for retrieval: ",
    query_instruction="Represent the question for retrieving supporting texts from the book passage: "
    )

# load vector store to use with langchain
docsearch = FAISS.load_local(folder_path=target_path, embeddings=embeddings)

# similarity search
question = "Who is big brother?"
search = docsearch.similarity_search(question, k=4)

for item in search:
    print(item.page_content)
    print(f"From page: {item.metadata['page']}")
    print("---")

# text = "This is a test document."
# query_result = embeddings.embed_query(text)
# doc_result = embeddings.embed_documents([text])
# print(doc_result)
# embeddings = HuggingFaceInstructEmbeddings(
#     query_instruction="Represent the query for retrieval: "
# )
# text = "This is a test document."
# query_result = embeddings.embed_query(text)

