import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

from prompts import code_mapping_extract
from vector.vector_store import get_vector_store

load_dotenv(".env")

api_docs_db = get_vector_store(dataset_name="met_office_api_docs")
open_ai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=open_ai_api_key,
)
code_extract_prompt = code_mapping_extract.get_prompt()
code_extract_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=api_docs_db.as_retriever(),
    chain_type_kwargs={"prompt": code_extract_prompt},
    verbose=True,
)

# Transform with LLM
with get_openai_callback() as cb:
    response = code_extract_chain.run(code_mapping_extract.question)
    print(response)
    print(cb)


# Store transformed data in vector store
code_mappings_db = get_vector_store(dataset_name="met_office_code_mappings")
document = Document(page_content=response)

# Store the entire document (no splitting as it's small)
print("Deleting documents in vector store")
code_mappings_db.delete(delete_all=True)
print("Storing document in vector store")
code_mappings_db.add_documents([document])
