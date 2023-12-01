import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('BACKEND_OPENAI_API_KEY')
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2,openai_api_key=OPENAI_API_KEY)

base_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, "web_search")
# 生成一个嵌入向量
vector = embeddings.embed_query("我该如何从台北去上海?")

