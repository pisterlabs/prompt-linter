from langchain.schema import Document
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pinecone

pinecone.init(api_key="c869cafc-6f9a-4abf-b8ca-d24ebc2f6ccd", environment="us-west1-gcp-free")

# vectorstore = Pinecone.from_existing_index(index_name="agent", embedding=OpenAIEmbeddings())

docs = []

# with open("data.txt", "r") as file:
#     text = file.read()
#     texts = text.split("\n\n")
#
#     for t in texts:
#         first_line = t.split("\n")[0]
#         rest = "\n".join(t.split("\n")[1:])
#
#         docs.append(Document(page_content=rest.strip(), metadata={"service": first_line.strip()}))
c = 0
metadata = ["details", "FAQs"]
with open("faqs.txt", "r") as file:
    text = file.read()
    texts = text.split("\n\n\n")
    for t in texts:
        docs.append(Document(page_content=t.strip(), metadata={"info": metadata[c]}))
        c += 1

print(len(docs))



vectorstore = Pinecone.from_documents(docs, embedding=OpenAIEmbeddings(), index_name="agent")