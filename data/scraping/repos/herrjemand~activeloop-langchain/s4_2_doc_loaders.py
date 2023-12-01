from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.document_loaders import TextLoader

loader = TextLoader("example.txt")
documents = loader.load()

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("2103.15348.pdf")
pages = loader.load_and_split()

from langchain.document_loaders import SeleniumURLLoader

urls = [
    "https://www.youtube.com/watch?v=TFa539R09EQ&t=139s",
    "https://www.youtube.com/watch?v=6Zv6A_9urh4&t=112s"
]

loader = SeleniumURLLoader(urls=urls, browser="firefox")
data = loader.load()

print(data[0])