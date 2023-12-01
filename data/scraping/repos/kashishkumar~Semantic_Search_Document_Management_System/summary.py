from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    get_response_synthesizer,
)
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.llms import OpenAI
import openai

openai.api_key = ""


chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=chatgpt, chunk_size=1024)

response_synthesizer = get_response_synthesizer(
    response_mode= "tree_summarize" , use_async=True
)

from pathlib import Path

# Get names of all files in the data directory as a list of strings
document_titles = [str(path) for path in Path("data").glob("*")]

documents = SimpleDirectoryReader("data").load_data()

documents_ = []
for document_title in document_titles:
    documents = SimpleDirectoryReader(input_files=[document_title]).load_data()
    documents[0].doc_id = document_title
    documents_.extend(documents)

doc_summary_index = DocumentSummaryIndex.from_documents(
    documents_,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
    show_progress=True
)


len(doc_summary_index.get_document_summary("data/Importer Agreement -Pudu&ST logistic track .pdf"))

summary_query_engine = doc_summary_index.as_query_engine(response_mode = "tree_summarize", use_async=True)

response = summary_query_engine.query("What is the date for importer agreement?")
print(response)