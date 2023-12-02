from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()
index_name = "gpt-test-pdf"
vectorstore = Pinecone.from_existing_index(index_name, embeddings,namespace='qingang')
query = "为什么不当部长"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)

