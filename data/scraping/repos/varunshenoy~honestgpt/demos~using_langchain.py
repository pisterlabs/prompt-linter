from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# read text and embed file
with open('texts/coffee.txt') as f:
    coffee_text = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(coffee_text)

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(texts, embeddings, metadatas=[
                              {"source": str(i)} for i in range(len(texts))])

# query and run chain
query = "What is the difference between a cappuccino and a latte?"
docs = docsearch.similarity_search(query)

chain = load_qa_with_sources_chain(OpenAI(temperature=0.7), chain_type="stuff")
result = chain({"input_documents": docs, "question": query},
               return_only_outputs=False)

print(result)
