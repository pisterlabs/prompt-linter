import myconfig
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader

loader = TextLoader("./data/ecommerce_faq.txt", encoding="utf-8")
documents = loader.load()

text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
# text_splitter = SpacyTextSplitter(chunk_size=256)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

faq_chain = VectorDBQA.from_chain_type(llm=OpenAI(temperature=0), vectorstore=vectorstore, verbose=True)


question = "请问你们的货，能送到三亚吗？大概需要几天？"
result = faq_chain.run(question)
print(result)