import os
from dotenv import load_dotenv
# Use the environment variables to retrieve API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Simple LLM call with generic knowledge
llm = OpenAI(model_name="text-davinci-003")
our_query = "What is the age of huhu"
print(llm(our_query))

# LLM call with our PDF as reference
data = PdfReader('Test.pdf')

combined_text = ''
for i, page in enumerate(data.pages):
  text = page.extract_text()
  if text:
    combined_text += text
combined_text

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=200,
    chunk_overlap=20,
    length_function = len,
)

finalData = text_splitter.split_text(combined_text)
len(finalData)

embeddings = OpenAIEmbeddings()

documentsearch = FAISS.from_texts(finalData, embeddings)
chain = load_qa_chain(OpenAI(),chain_type="stuff")

our_query = "Who is naval"
docs = documentsearch.similarity_search(our_query)
print(chain.run(input_documents=docs, question= our_query))