from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from chromadb.config import Settings

import os
import sys
import argparse
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

CHROMA_SETTINGS = Settings(
  chroma_db_impl='duckdb+parquet',
  persist_directory="chroma_persist",
  anonymized_telemetry=False
)

def build_knowledge():
  # embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
  embeddings = LlamaCppEmbeddings(model_path="../infinite_website/models/llama-7b.ggmlv3.q2_K.bin")
  chunk_size = 500
  chunk_overlap = 50

  documents = []

  for root, dirs, files in os.walk("mermaid_docs"):
    for file in files:
      file_path = os.path.join(root, file)
      if not file_path.endswith(".md"):
        continue
      print(file_path)
      documents.extend(UnstructuredMarkdownLoader(file_path).load())

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  texts = text_splitter.split_documents(documents)

  print("Creating embeddings...")
  db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_persist", client_settings=CHROMA_SETTINGS)
  db.persist()
  db = None

def ask_with_memory(line):
  # embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
  embeddings = LlamaCppEmbeddings(model_path="../infinite_website/models/llama-7b.ggmlv3.q2_K.bin")

  db = Chroma(persist_directory="chroma_persist", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

  retriever = db.as_retriever()

  res = ""
  # llm = ChatOpenAI(temperature=0)
  callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
  llm = llm = LlamaCpp(
    model_path="../infinite_website/models/llama-7b.ggmlv3.q2_K.bin",
    callback_manager=callback_manager,
    verbose=True
  )
  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

  # Get the answer from the chain
  res = qa("---------------------\nYou are a hyperintelligent software engineer. Using the documentation provided, assist with the following problem:\nQuestion: " + line + "\nResponse:")
  answer, docs = res['result'], res['source_documents']
  res = answer + "\n\n\n" + "Sources:\n"
  
  sources = set()  # To store unique sources
  
  # Collect unique sources
  for document in docs:
    if "source" in document.metadata:
      sources.add(document.metadata["source"])
  
  # Print the relevant sources used for the answer
  for source in sources:
    if source.startswith("http"):
      res += "- " + source + "\n"
    else:
      res += "- source code: " + source + "\n"
  
  return res

if __name__ == "__main__":
  build_knowledge()

  parser = argparse.ArgumentParser("simple-summarizer")
  parser.add_argument("-f", "--file", help="The file to summarize.")
  args = parser.parse_args()

  # Try to read the file's content into file_content
  file_content = ""
  try:
    with open(args.file, 'r') as f:
      lines = f.readlines()
      for i, line in enumerate(lines, start=1):
        file_content += f"{str(i).rjust(6, '0')} {line}"
  except FileNotFoundError:
    sys.stderr.write("Please supply a filename with the flag '--file' or '-f'.")
    exit()
  except IOError:
    sys.stderr.write("Error reading the file.")
    exit()

#  Use the line numbers instead of the content of each line.
  # An example of turning code into a mermaid.js diagram is given below:
  # --- begin example.py ---
  # while True:
  #   print('hello!')
  # --- end example.py ---
  # --- begin diagram ---
  # flowchart LR
  #   A["`while True:`"]
  #   B["`print('hello!')`"]
  #   A --> B
  # --- end diagram ---

  prompt = f"""
Now, generate a class diagram for the following code in the format mermaid specifies:
--- begin {args.file} ---
{file_content}
--- end {args.file} ---
  """

  print(prompt)
  # exit()

  # print(ask_with_memory(prompt))
  # while True:
  #   x =  input("> ") 
  #   inp = []
  #   while x != '':  
  #       inp.append(x) 
  #       x = input("> ")
    
  #   print(ask_with_memory("\n".join(inp)))


  # while True:
  #   print(ask_with_memory(input("> ")))

  print(ask_with_memory(prompt))
