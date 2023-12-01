from langchain.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
import pyttsx3
import pinecone
import os
engine = pyttsx3.init()
voices = engine.getProperty('voices')

load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
loader = UnstructuredEPubLoader('../elon_mask.epub')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3200,chunk_overlap=500)
texts = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)
# 创建一个向量存储库
vectorstore = Pinecone.from_texts(
    [t.page_content for t in texts],
    embeddings,
    index_name='paper')

# 初始化OpenAI大语言模型
llm = OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm=llm,chain_type="map_reduce",
                      return_intermediate_steps=True, 
                      )
# 问题
query = "spaceX在研究过程中遇到哪些困难,请用中文回答"

# 在向量数据中搜索相似文档，返回文档列表
# 你可以通过设置num_results来控制返回的文档数量,默认为4
docs = vectorstore.similarity_search(query, k=1)
origin_text = docs[0].page_content
if(len(origin_text)>1800):
    docs[0].page_content = origin_text[:1800]
result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
print(result['output_text'])

engine.say(result['output_text'])
engine.runAndWait()
