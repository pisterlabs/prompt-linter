from langchain.document_loaders import UnstructuredExcelLoader
loader = UnstructuredExcelLoader("../query2.xlsx")
docs = loader.load()
print(docs[0].page_content)
