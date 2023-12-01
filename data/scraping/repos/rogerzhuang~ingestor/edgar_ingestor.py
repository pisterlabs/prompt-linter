from ingestors.ingestor import Ingestor
from langchain.vectorstores import Pinecone
# from langchain.vectorstores import Milvus
# from langchain.vectorstores import Weaviate
# from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import tiktoken


class EDGARIngestor(Ingestor):
    def __init__(self, file_path, props, embeddings):
        super().__init__(file_path, embeddings)
        self.props = props

    def get_documents(self):
        loader = UnstructuredHTMLLoader(self.file_path)
        doc = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=lambda text: len(tiktoken.get_encoding(
                'cl100k_base').encode(text, disallowed_special=())),
            separators=['\n\n', '\n', ' ', '']
        )
        documents = []
        chunks = text_splitter.split_text(doc[0].page_content)
        for i, chunk in enumerate(chunks):
            document = Document(
                page_content=chunk,
                metadata={
                    "symbol": self.props['prop_2'],
                    "form_type": self.props['prop_3'],
                    "report_date": int(self.props['prop_4']),
                    "source": 'https://www.sec.gov/Archives/edgar/data/' + self.props['file_name'].replace(',', '/')
                },
            )
            documents.append(document)
        return documents

    def ingest(self):
        documents = self.get_documents()
        Pinecone.from_documents(
            documents, self.embeddings, index_name=os.environ['PINECONE_INDEX_EDGAR'])
        # vector_db = Chroma.from_documents(
        #     documents, self.embeddings, collection_name='edgar', client=self.chroma_client)
        print(
            f"Successfully sent {len(documents)} documents of symbol ({self.props['prop_2']}) form_type ({self.props['prop_3']}) report_date ({self.props['prop_4']}) to Pinecone.")
        # print(f"Successfully sent {len(documents)} documents to Chroma.")
