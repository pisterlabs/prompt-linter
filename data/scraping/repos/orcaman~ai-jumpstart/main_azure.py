import sys
import dotenv

dotenv.load_dotenv()

import nltk
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA


class PDFAnalyzer:
    def __init__(self, pdf_file, query):
        self.pdf_file = pdf_file
        self.query = query
        self.documents = None
        self.texts = None
        self.chain = None

    @staticmethod
    def download_nltk_data():
        nltk.download("punkt")

    def load_pdf_file(self):
        loader = UnstructuredFileLoader(self.pdf_file)
        self.documents = loader.load()

    def split_documents_into_chunks(self, chunk_size=800, chunk_overlap=0):
        UnstructuredFileLoader(self.documents, mode='elements')
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.texts = text_splitter.split_documents(self.documents)

    def prepare_model_embedding(self):
        embeddings = OpenAIEmbeddings(
            chunk_size=1,
            deployment='text-embedding-ada-002',
        )
        doc_search = Chroma.from_documents(self.texts, embeddings)
        llm = AzureOpenAI(deployment_name='text-davinci-003', model_name="text-davinci-003")
        self.chain = RetrievalQA.from_chain_type(llm=llm, retriever=doc_search.as_retriever())

    def analyze_pdf(self):
        self.download_nltk_data()
        self.load_pdf_file()
        self.split_documents_into_chunks()
        self.prepare_model_embedding()

        print('\n\n\n\n\n-----------------')
        print('question:', self.query)
        print('answer:', self.chain.run(self.query))
        print('-----------------\n\n')


def main():
    if len(sys.argv) < 3:
        print("Missing arguments: pdf file and/or query")
        sys.exit(1)

    pdf_file = sys.argv[1]
    query = sys.argv[2]

    analyzer = PDFAnalyzer(pdf_file, query)
    analyzer.analyze_pdf()


if __name__ == "__main__":
    main()
