from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
os.environ["OPENAI_API_KEY"] = "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"

pdfreader = PdfReader('interview.pdf')

from typing_extensions import Concatenate
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# print(raw_text)

# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

document_search = FAISS.from_texts(texts, embeddings)
chain = load_qa_chain(OpenAI(), chain_type="stuff")


query = "Ask me any one of the question."
docs = document_search.similarity_search(query)
print(chain.run(input_documents=docs, question=query))
