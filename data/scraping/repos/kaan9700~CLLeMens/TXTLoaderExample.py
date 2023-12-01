from CLLeMensLangchain.loaders.text_loader import TxtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


content = TxtLoader(file_path="../../media/uploads/Lorem.txt")
pages = content.load()

