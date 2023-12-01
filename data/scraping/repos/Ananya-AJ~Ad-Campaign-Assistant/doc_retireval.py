import glob
import json

from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI

pdf_folder_path = 'Data/pdfs'
model_path = 'sentence-transformers/all-MiniLM-l6-v2'

API_KEY = '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'
SIMILARITY_THRESHOLD = 1.0 # threshhold to determine if product query is successful or not
PROMPT_TEMPLATES_FILEPATH = 'Data/prompt_templates.json'

model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}

loader = PyPDFDirectoryLoader(pdf_folder_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db = FAISS.from_documents(all_splits, embeddings)

# store this index
db.save_local("product_index")
db = FAISS.load_local("product_index", embeddings)

# question = "what are the features of a smart watch for fitness enthusiasts?"
# product_doc = db.similarity_search_with_score(question, k=1)
def summarize_product_doc(file):
    prompt_template = """
      For this product documentation, write a summary that will be useful for the marketing team to advertise the product. Keep as many details as possible.
      "{text}"
      SUMMARY:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo-16k", openai_api_key=API_KEY, request_timeout=120) # if timeout is less, no output
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    print("Creating product document summary")
    loader = PyPDFLoader(file)
    docs = loader.load()
    summary_result = stuff_chain.run(docs)

    return summary_result

def create_ad_prompt(
    product_query: str,
    user_strategy: str,
    user_structure: str,
    user_context: str,
) -> tuple[bool, str]:

  # search for the doc in the db
  product_doc = db.similarity_search_with_score(product_query, k=1)

  product_file = product_doc[0][0].metadata["source"]
  product_found = product_doc[0][1] <= SIMILARITY_THRESHOLD

  print("Product found") if product_found else print("Product not found")

  with open('Data/prompt_templates.json', 'r') as f:
    prompt_sections = json.load(f)
  print(prompt_sections)
  prompt_part_1 = prompt_sections["strategy"][user_strategy]
  prompt_part_2 = prompt_sections["structure"][user_structure]
  prompt_part_3 = prompt_sections["context"][user_context]
  prompt_part_4 = summarize_product_doc(product_file) if product_found else ""

  final_prompt = "\n".join([prompt_part_1, prompt_part_2, prompt_part_3, prompt_part_4])

  return product_found, final_prompt

# product_query = "what are the features of a smart watch for fitness enthusiasts?"
# user_strategy = "Festive Promotion"
# user_structure = "Problem solution"
# user_context = "Refer to"

# product_found, final_prompt = create_ad_prompt(product_query, user_strategy, user_structure, user_context)

