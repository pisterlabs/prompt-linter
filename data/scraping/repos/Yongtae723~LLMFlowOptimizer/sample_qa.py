from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.document_loaders import TextLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.text_splitter import TextSplitter


class SampleQA:
    """Define the flow of the model to be adjusted."""

    def __init__(
        self,
        data_path: str,
        embedding: Embeddings,
        text_splitter: TextSplitter,
        llm: BaseLanguageModel,
    ) -> None:
        """Input the elements necessary for LLM flow The arguments here will be used as a
        hyperparameters and optimized.

        the arguments are defined by `configs/model/sample.yaml`
        """
        self.embedding = embedding
        self.text_splitter = text_splitter
        self.text_loader = TextLoader(data_path)
        self.llm = llm
        self.index = VectorstoreIndexCreator(
            embedding=self.embedding, text_splitter=self.text_splitter
        ).from_loaders([self.text_loader])

        self.chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.index.vectorstore.as_retriever(),
            return_source_documents=True,
        )

    def __call__(self, question: str) -> str:
        """Answer the question."""
        return self.chain(question)

    def get_chain(self) -> Chain:
        """Get langchain chain."""
        return self.chain
