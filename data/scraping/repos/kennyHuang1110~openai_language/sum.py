from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

# 导入文本
loader = UnstructuredFileLoader("/content/sample_data/data/" ,glob='**/*.txt')
# 将文本转成 Document 对象
documents = loader.load()
print(f'documents:{len(documents)}')

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0
)

# 切分文本
split_documents = text_splitter.split_documents(documents)
print(f'documents:{len(split_documents)}')

# 加载 llm 模型
llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=1500)

# 创建总结链
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# 执行总结链，（为了快速演示，只总结前5段）
chain.run(split_documents[:5])
print(chain.run(split_documents))

