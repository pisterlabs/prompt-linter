import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在目录的上级目录路径
parent_directory = os.path.dirname(current_file_path)
parent_parent_directory = os.path.dirname(parent_directory)

# 拼接文件路径
pdf_path = os.path.join(parent_parent_directory, "Datasets/Papers/CoT/2201.11903.pdf")

loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
print(f"加载完毕，共加载{len(pages)}页PDF文件")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10,
    length_function=len
)

texts = text_splitter.split_documents(pages)
print(f"文章共切分为{len(texts)}段")
for i in range(0, len(texts)):
    print(f"段{i}")
    print(texts[i].page_content)
