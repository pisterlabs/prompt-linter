from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

url = "https://en.wikipedia.org/wiki/Hwa_Chong_Institution"

loader = WebBaseLoader(url).load()
count = 0

output_file = open("output.txt", "w")
output_file.write(loader[0].page_content.replace("\n", " "))

with open("output.txt", "r") as file:
    hwa_chong_info = file.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 20000,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)