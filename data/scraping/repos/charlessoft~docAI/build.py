import os
import logging
import sys

from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import openai
from langchain.llms import AzureOpenAI

os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_API_KEY'

openai.api_type = "azure"
openai.api_base = "https://adt-openai.openai.azure.com/"
openai.api_version = "2022-12-01"
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY", '938ce9d50df942d08399ad736863d063')

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://adt-openai.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "938ce9d50df942d08399ad736863d063"

OPENAI_API_KEY="938ce9d50df942d08399ad736863d063"
PINECONE_API_KEY="33e67396-4ede-4259-b084-73f5cd10098d"
PINECONE_API_ENV="us-east4-gcp"


# 读取data文件夹下的文档
documents = SimpleDirectoryReader('../data').load_data()
# 按最大token数500来把原文档切分为多个小的chunk，每个chunk转为向量，并构建索引
index = GPTSimpleVectorIndex.from_documents(documents)
# 保存索引
index.save_to_disk('index.json')
