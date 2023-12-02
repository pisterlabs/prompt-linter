import os
from typing import List
from timeit import default_timer as timer
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Constants
gpt4all_j_path = "../../models/ggml-gpt4all-j.bin"
index_path = "../data/chromadb/"
docs_path = "../data/docs/"

## utility functions

import textwrap
import os


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split("\n")

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = "\n".join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response["answer"]))
    print("\nSources:")
    for source in llm_response["source_documents"]:
        print("  URL: " + str(source.metadata["url"]))


def load_documents(loading_all) -> List:
    loader = DirectoryLoader(
        docs_path, glob="**/*.html", loader_cls=UnstructuredHTMLLoader
    )
    all_docs = loader.load()

    urls = [
        "https://www.priceless.com/m/filter/options/category/506",
        # "https://www.priceless.com/m/filter/options/category/510",
        # "https://www.priceless.com/m/filter/options/location/9716/trk/20211/",
    ]
    documents = []

    for doc in all_docs:
        src = doc.metadata["source"]
        url = (
            src.replace("../data/docs/", "https://")
            .replace("index.html", "")
            .replace(".html", "")
        )
        if not loading_all and url not in urls:
            continue
        doc.metadata = dict()
        doc.metadata["url"] = url
        documents.append(doc)

    return documents


def split_chunks(documents: List) -> List:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    return text_splitter.split_documents(documents)


def generate_index(chunks: List, embeddings: HuggingFaceInstructEmbeddings) -> Chroma:
    chromadb_instructor_embeddings = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=index_path
    )

    chromadb_instructor_embeddings.persist()
    return chromadb_instructor_embeddings


# Main execution
# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
# Verbose is required to pass to the callback manager
# llm = GPT4All(model=gpt4all_j_path, callbacks=callbacks, verbose=True)
# If you want to use GPT4ALL_J model add the backend parameter
llm = GPT4All(
    model=gpt4all_j_path, n_ctx=2048, backend="gptj", callbacks=callbacks, verbose=True
)
print("\nDONE\n")

embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
)
print("DONE\n")

if not os.path.isdir(index_path):
    print("The index persist directory is not present. Creating a new one.")
    os.mkdir(index_path)
    sources = load_documents(False)
    print(f"Splitting {len(sources)} files ...")

    chunks = split_chunks(sources)
    print(f"Generating index for {len(chunks)} chunks ...")

    index = generate_index(chunks, embeddings)
else:
    print("The index persist directory is present. Loading index ...")
    index = Chroma(embedding_function=embeddings, persist_directory=index_path)

print("DONE\n")

qa = ConversationalRetrievalChain.from_llm(
    llm,
    index.as_retriever(search_kwargs={"k": 4}),
    max_tokens_limit=400,
    return_source_documents=True,
)

# Chatbot loop
chat_history = []
print("Welcome to the Priceless Chatbot! Type 'exit' to stop.")
while True:
    query = input("Please enter your question: ")

    if query.lower() == "exit":
        break

    print("\nQuestion: " + query)

    start = timer()
    result = qa({"question": query, "chat_history": chat_history})
    end = timer()

    process_llm_response(result)
    print(f"Completed in {end - start:.3f}s")
