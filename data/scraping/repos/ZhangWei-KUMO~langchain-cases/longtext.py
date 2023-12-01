from langchain.document_loaders import UnstructuredFileLoader
import nltk
nltk.download('punkt')
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

# 导入文本
loader = UnstructuredFileLoader("/Users/lewiszhang/Desktop/quant.txt")
# 将文本转成 Document 对象
document = loader.load()
print(f'documents:{len(document)}')

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 500,
#     chunk_overlap = 0
# )
# split_documents = text_splitter.split_documents(document)
# print(f'documents:{len(split_documents)}')

# # 加载 llm 模型
# llm = OpenAI(model_name="gpt-3.5-turbo-0301", max_tokens=1000)

# # 创建总结链
# chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# # 执行总结链，（为了快速演示，只总结前5段）
# chain.run(split_documents[:5])