import myconfig
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI
from langchain.document_loaders import TextLoader

loader = TextLoader("./data/ecommerce_faq.txt", encoding="utf-8")
documents = loader.load()

text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
# text_splitter = SpacyTextSplitter(chunk_size=256)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever())
question = "请问你们的货，能送到三亚吗？大概需要几天？"
result = qa.run(question)

print(result)
