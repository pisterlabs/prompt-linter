import json
import chromadb
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from progress.bar import IncrementalBar


def count_strings_in_range(string_array, max_length):
    count_array = [0] * ((max_length - 0) + 1)

    for string in string_array:
        length = len(string)
        if 0 <= length <= max_length:
            count_array[length - 0] += 1

    return count_array


class Embedder:
    def __init__(self, dataset, persist_directory):
        self.chunks = []
        self.dataset = dataset
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet", persist_directory=persist_directory
            )
        )

    def preprocess(self, min_chunk_size, max_chunk_size, chunk_overlap):
        with open(self.dataset) as r:
            docs_raw = list(json.loads(r.read()))

            docs_clean = []
            for e in docs_raw:
                if "text" not in e:
                    continue
                if not e["text"]:
                    continue
                if e["text"] == "":
                    continue
                if len(e["text"]) < min_chunk_size:
                    continue
                docs_clean.append(e)
            print(f"\n* Filtered {len(docs_raw)} sources down to {len(docs_clean)}")
            cleaned_docs = [
                Document(
                    page_content=doc["text"],
                    metadata={
                        "id": doc["id"],
                        "title": doc.get("title", ""),
                    },
                )
                for doc in docs_clean
            ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        self.chunks = [
            (str(index), chunk)
            for index, chunk in enumerate(text_splitter.split_documents(cleaned_docs))
            if len(chunk.page_content) > min_chunk_size
        ]
        # plot_hist(
        #     [len(chunk[1].page_content) for chunk in self.chunks],
        #     bincount=64,
        #     binwidth=8,
        #     xlab="Length of chunk",
        #     showSummary=True,
        # )
        return self

    def embed(self, name, embedding_function):
        collection = self.client.create_collection(
            name, embedding_function=embedding_function
        )

        with IncrementalBar(
            f"* Embedding {len(self.chunks)} chunks",
            suffix="%(percent).1f%% - %(elapsed)ds",
            max=len(self.chunks),
        ) as bar:
            for chunk in self.chunks:
                collection.add(
                    ids=chunk[0],
                    documents=chunk[1].page_content,
                    metadatas={
                        "id": chunk[1].metadata["id"],
                        "title": chunk[1].metadata["title"],
                    },
                )
                bar.next()
        print("* Finished embedding documents")


embed_func = embedding_functions.InstructorEmbeddingFunction(
    model_name="hkunlp/instructor-large", device="cpu"
)

embedder = Embedder("local/data.json", "local/embeddings")
embedder.preprocess(120, 1024, 128)
embedder.embed("general-min_chunk_size", embed_func)
