import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


class SentenceTransformerEmbeddings:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/all-mpnet-base-v2',
        )
        self.model = AutoModel.from_pretrained(
            'sentence-transformers/all-mpnet-base-v2',
        )
        return

    def embed_fn(self, sentences: list[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        return self.mean_pooling(
            model_output,
            encoded_input['attention_mask'],
        )

    def embed_documents(
            self,
            documents: list[str],
    ) -> list[list[float]]:
        return self.embed_fn(documents).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_fn([query]).tolist()[0]

    # Mean Pooling - Take attention mask into account for correct averaging
    @staticmethod
    def mean_pooling(
            model_output: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (attention_mask
                            .unsqueeze(-1)
                            .expand(token_embeddings.size())
                            .float())
        return (
            torch.sum(token_embeddings * input_mask_expanded, 1) /
            torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )


embed_fn = SentenceTransformerEmbeddings()


if not os.path.exists('data/om_embeddings'):
    loader = DirectoryLoader(
        './data/ModelicaStandardLibrary/Modelica/',
        glob='**/[!package]*.mo',
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
    )

    docs = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=0,
        separators=[
            '\nmodel ',
            '\nblock ',
            '\nfunction ',
            # '\nannotation',
            # '\n\n',
        ],
    ).split_documents(loader.load())

    chunk = 100
    db = Chroma.from_documents(
        docs[:chunk],
        embed_fn,
        persist_directory='./data/om_embeddings',
    )

    for begin, end in tqdm(zip(
            range(chunk, len(docs), chunk),
            range(chunk * 2, len(docs), chunk),
    ), total=len(docs) // chunk):
        db.add_texts(
            texts=[doc.page_content for doc in docs[begin:end]],
            metadatas=[doc.metadata for doc in docs[begin:end]],
        )
    db.persist()

else:
    db = Chroma(
        embedding_function=embed_fn,
        persist_directory='./data/om_embeddings',
    )
