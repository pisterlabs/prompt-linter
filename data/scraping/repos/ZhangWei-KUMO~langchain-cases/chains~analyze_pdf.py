from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import os
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=1000)
loader = UnstructuredPDFLoader('../paper.pdf')
book = loader.load()
summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
result = summarize_document_chain.run(book)
print(result)