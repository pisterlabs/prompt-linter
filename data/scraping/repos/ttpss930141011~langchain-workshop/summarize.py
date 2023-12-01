from dotenv import dotenv_values
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")

# 導入文本
loader = UnstructuredFileLoader("./data/text/lg_test.txt")
# 將文本轉成 Document 對象
document = loader.load()
print(f'documents:{len(document)}')

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

# 切分文本
split_documents = text_splitter.split_documents(document)
print(f'documents:{len(split_documents)}')

# 加載 llm 模型
llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)

# 創建總結鏈
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# 執行總結鏈，（為了快速演示，只總結前5段）
chain.run(split_documents[:5])