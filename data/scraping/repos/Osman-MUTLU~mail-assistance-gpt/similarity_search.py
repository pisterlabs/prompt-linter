import time
import os


from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

file_paths = os.listdir("mails/b0b04c690b13d296")
file_paths = ["mails/b0b04c690b13d296/"+file_path for file_path in file_paths]
data = []
for file_path in file_paths:
    loader = TextLoader(file_path=file_path)
    data.append(loader.load()[0])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
question = "Batuhan bana mail yolladı mı?"
docs = vectorstore.similarity_search(question.lower(), top_k=2)
context = ""
for doc in docs:
    context += "-------------mail----------------\n"
    context += doc.page_content[:1000]
    context += "\n\n"

print(context)


from gpt4all import GPT4All
model = GPT4All(model_name="ggml-vicuna-13b-1.1-q4_2.bin",model_path="models")

prompt = f"""
"Summarize the main themes in these retrieved mails: 
{context}"
---
Question: {question}
Answer: """

with model.chat_session():
    result = model.generate(prompt, max_tokens=2048, top_k=1)
    print(result)