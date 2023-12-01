from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
#normal prompt
# from langchain.llms import OpenAI
# llm = OpenAI(model_name="text-davinci-003")
# # print(llm("explain what a box office total is."))

# # gaslight prompt
# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )
# from langchain.chat_models import ChatOpenAI
# chat = ChatOpenAI(model_name="gpt-3.5-turbo")
# messages = [
#     SystemMessage(content="You are an award winning film director"),
#     HumanMessage(content="What was your best movie?")
# ]
# # response = chat(messages)


# #variable input prompt
# from langchain import PromptTemplate
# template = '''
# You are an expert data scientist. Explain this concept, {concept}
# '''
# prompt = PromptTemplate(
#     input_variables=["concept"],
#     template=template
# )
# print(llm(prompt.format(concept="machine learning")))

# #chains
# from langchain.chains import LLMChain
# from langchain.chains import SimpleSequentialChain
# chain = LLMChain(llm=llm, prompt=prompt)
# chain.run("machine learning")

# secondPrompt = PromptTemplate(
#     input_variables=["concept"],
#     template="Take the description of {concept} and explain it to a 5 year old."
# )
# chain2 = LLMChain(llm=llm, prompt=secondPrompt)

# totalChain = SimpleSequentialChain(
#     chains=[chain, chain2],
#     verbose=True
# )
# explain = totalChain.run("machine learning")
# print(explain)

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# textSplitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     overlap=0
# )
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def extract_text_from_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ""
    for page in range(len(pdf.pages)):
        text += pdf.pages[page].extract_text()
    return text

text = extract_text_from_pdf("TRUE BETRAYALS.pdf")
print(text[:500]) # Print the first 500 characters to check the result.

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 30000,
    chunk_overlap  = 0,
    length_function = len,
)
texts = text_splitter.split_text(text)
embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)
llm = OpenAI(model_name="gpt-4")
chain = load_qa_chain(llm=llm, chain_type="stuff")
query = "What is the plot of the story?"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)