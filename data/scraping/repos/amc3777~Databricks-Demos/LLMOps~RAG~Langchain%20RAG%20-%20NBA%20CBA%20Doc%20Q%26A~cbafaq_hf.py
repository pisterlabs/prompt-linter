# Databricks notebook source
# MAGIC %sh
# MAGIC wget -O cba.pdf https://imgix.cosmicjs.com/25da5eb0-15eb-11ee-b5b3-fbd321202bdf-Final-2023-NBA-Collective-Bargaining-Agreement-6-28-23.pdf

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install pypdf faiss-cpu langchain transformers sentence_transformers

# COMMAND ----------

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("cba.pdf")
pages = loader.load_and_split()

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

faiss_index = FAISS.from_documents(pages, embedding_model)

# COMMAND ----------

def get_similar_docs(question, similar_doc_count):
  return faiss_index.similarity_search(question, k=similar_doc_count)

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

def build_qa_chain():
  torch.cuda.empty_cache()
  model_name = "databricks/dolly-v2-12b"

  instruct_pipeline = pipeline(model=model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", 
                               return_full_text=True, max_new_tokens=100)

  template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  Instruction: 
  You are an NBA salary cap expert and your job is to provide the best answer based on the latest NBA collective bargaining agreement. 
  Use only information in the following paragraphs to answer the question at the end. Explain the answer with reference to these paragraphs. If you do not know, say that you do not know.

  {context}
 
  Question: {question}

  Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

  hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)

  return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt)

# COMMAND ----------

qa_chain = build_qa_chain()

# COMMAND ----------

def answer_question(question):
  similar_docs = get_similar_docs(question, similar_doc_count=2)
  result = qa_chain({"input_documents": similar_docs, "question": question})
  return result['output_text']

# COMMAND ----------

answer_question("What is a non-simultaneous trade?")

# COMMAND ----------

from pypdf import PdfReader

reader = PdfReader("cba.pdf")
number_of_pages = len(reader.pages)
i = 24
text = ""

while i < number_of_pages:
  page = reader.pages[i]
  text += page.extract_text()
  i += 1

print(str(len(text)))
