
from langchain.document_loaders import PagedPDFSplitter

loader = PagedPDFSplitter("data\pdf\土木技術管理規程集_道路Ⅰ編.pdf")


pages = loader.load_and_split()
print(pages[50].page_content)

 