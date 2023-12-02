from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=4000)
loader = TextLoader('../20report.txt')
book = loader.load()
summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
print(summary_chain)
summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
result = summarize_document_chain.run(book)
# print(result)