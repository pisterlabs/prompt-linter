import marqo
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Marqo as Vector
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd
import config


class FilePocessor:
    def __init__(self, fileLocation):
        self.fileLocation = fileLocation
        self.client = marqo.Client(config.marqo_url)
        self.index_name = config.index_name

    def process(self, contentType):
        match contentType:
            case "text/plain":
                documents = self.loadTxt()
            case "application/pdf":
                documents = self.loadPdf()
            case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                documents = self.loadDocx()
            case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                self.fileLocation = self.__convertExcelToCsv()
                documents = self.loadCsv()
            case "text/csv":
                documents = self.loadCsv()
            case _:
                return "Unsupported file type"

        text_splitter = CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        docs = text_splitter.split_documents(documents)
        index = self.__indexDocument(docs)

        return index

    def loadPdf(self):
        loader = PyPDFLoader(self.fileLocation)
        return loader.load_and_split()

    def loadTxt(self):
        loader = TextLoader(self.fileLocation)
        return loader.load()

    def loadDocx(self):
        loader = Docx2txtLoader(self.fileLocation)
        return loader.load()

    def loadCsv(self):
        loader = CSVLoader(self.fileLocation)
        return loader.load()

    def __convertExcelToCsv(self):
        outputName = self.fileLocation.replace('xlsx', 'csv')
        data = pd.read_excel(self.fileLocation, engine='openpyxl')
        data.to_csv(outputName, index=False)
        return outputName

    def __indexDocument(self, docs):
        vectorstore = Vector(self.client, self.index_name)
        return vectorstore.add_documents(docs)
        # try:
        # except Exception:
        #     return "An exception occurred"
