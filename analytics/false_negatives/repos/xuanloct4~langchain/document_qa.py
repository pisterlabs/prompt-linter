

import environment

# from abc import ABC, abstractmethod
# from typing import List
# from langchain.schema import Document

# class BaseRetriever(ABC):
#     @abstractmethod
#     def get_relevant_documents(self, query: str) -> List[Document]:
#         """Get texts relevant for a query.

#         Args:
#             query: string to find relevant texts for

#         Returns:
#             List of relevant documents
#         """

# from langchain.chains import RetrievalQA
# from langchain.document_loaders import TextLoader
# loader = TextLoader('./documents/state_of_the_union.txt', encoding='utf8')

# from langchain.indexes import VectorstoreIndexCreator

# index = VectorstoreIndexCreator().from_loaders([loader])

# query = "What did the president say about Ketanji Brown Jackson"
# print(index.query(query))

# query = "What did the president say about Ketanji Brown Jackson"
# print(index.query_with_sources(query))

# index.vectorstore

# index.vectorstore.as_retriever()

# documents = loader.load()

# from langchain.text_splitter import CharacterTextSplitter
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# from langchain.embeddings import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings()

# from langchain.vectorstores import Chroma
# db = Chroma.from_documents(texts, embeddings)

# retriever = db.as_retriever()

# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
# query = "What did the president say about Ketanji Brown Jackson"
# print(qa.run(query))

# index_creator = VectorstoreIndexCreator(
#     vectorstore_cls=Chroma, 
#     embedding=OpenAIEmbeddings(),
#     text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# )








# query = "What did the president say about Ketanji Brown Jackson"
# index.query(query)

# query = "What did the president say about Ketanji Brown Jackson"
# index.query_with_sources(query)

# from langchain.chains.question_answering import load_qa_chain
# chain = load_qa_chain(llm, chain_type="stuff")
# chain.run(input_documents=docs, question=query)

# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# chain = load_qa_with_sources_chain(llm, chain_type="stuff")
# chain({"input_documents": docs, "question": query}, return_only_outputs=True)




#!pip install git+https://github.com/ooo27/langchain.git
# langchain/callbacks/streaming_stdout.py
# sys.stdout.write(token)
# sys.stdout.write(str(token, "UTF-8"))

# langchain/llms/gpt4all.py
#             text = self.client.generate(
#             text = self.client.cpp_generate(
#                 prompt,
#                 new_text_callback=text_callback,
#                 **self._default_params,
#             )
#         else:
#             text = self.client.generate(prompt, **self._default_params)
#             text = self.client.cpp_generate(prompt, **self._default_params)
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding
from langchain.document_loaders import TextLoader
loader = TextLoader("./documents/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

docsearch = Chroma.from_documents(texts, embedding)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
query = "What did the president say about Ketanji Brown Jackson"
print("---------------stuff query What did the president say about Ketanji Brown Jackson-----------------------")
print(qa.run(query))
print("---------------End stuff query What did the president say about Ketanji Brown Jackson-----------------------")

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=docsearch.as_retriever())
query = "What did the president say about Ketanji Brown Jackson"
print("---------------End map_reduce query What did the president say about Ketanji Brown Jackson-----------------------")
print(qa.run(query))
print("---------------End map_reduce query What did the president say about Ketanji Brown Jackson-----------------------")


# from langchain.chains.question_answering import load_qa_chain
# qa_chain = load_qa_chain(llm, chain_type="stuff")
# qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=docsearch.as_retriever())

# query = "What did the president say about Ketanji Brown Jackson"
# print(qa.run(query))


# from langchain.prompts import PromptTemplate
# prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {question}
# Answer in Italian:"""
# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )

# chain_type_kwargs = {"prompt": PROMPT}
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)

# query = "What did the president say about Ketanji Brown Jackson"
# print(qa.run(query))

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)
query = "What did the president say about Ketanji Brown Jackson"
result = qa({"query": query})
print("---------------Result-----------------------")
print(result["result"])
print(result["source_documents"])
print("---------------End Result-----------------------")

