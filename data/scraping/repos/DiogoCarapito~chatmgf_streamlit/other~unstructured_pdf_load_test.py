from langchain.document_loaders import UnstructuredPDFLoader

file = OnlinePDFLoader('http://nocs.pt/wp-content/uploads/2016/01/Programa-Nacional-Vigilancia-Gravidez-Baixo-Risco-2015.pdf')

loader = UnstructuredPDFLoader(file)

data = loader.load()