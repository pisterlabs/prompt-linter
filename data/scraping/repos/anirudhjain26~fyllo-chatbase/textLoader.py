from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# ENTER THE RELEVANT FILE NAME HERE (ONLY TEXT FILES LIKE .txt and .md), 
# For PDF files or CSV, check langchain Document Loader Documentation
loader = TextLoader("scrapedText.txt")

# ENTER YOUR QUERY HERE
query = "How can irrigation system companies benefit from it?"


pages = loader.load_and_split()
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(pages, embeddings).as_retriever()
docs = docsearch.get_relevant_documents(query)
chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
output = chain.run(input_documents=docs, question=query)
print(output)