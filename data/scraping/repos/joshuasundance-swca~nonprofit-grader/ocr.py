import os
# from azure.ai.formrecognizer import DocumentAnalysisClient
# from azure.core.credentials import AzureKeyCredential
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
# from langchain.document_loaders.pdf import DocumentIntelligenceLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient


# from pypdf import PdfReader, PdfWriter


def truncate_pdf_to_two_pages(input_path: str, output_path: str) -> None:
    # Read the source PDF
    reader = PdfReader(input_path)
    writer = PdfWriter()

    # Check if the PDF has more than two pages
    if len(reader.pages) > 2:
        # Add the first two pages to the writer
        writer.add_page(reader.pages[0])
        writer.add_page(reader.pages[1])

        # Write the truncated PDF to the new file
        with open(output_path, 'wb') as output_pdf:
            writer.write(output_pdf)
        print(f"Truncated: {input_path} -> {output_path}")
    else:
        print(f"No need to truncate (2 or less pages): {input_path}")

def truncate_pdfs_in_dir(root_dir: str) -> None:
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                output_path = os.path.join(root, f"truncated_{file}")
                truncate_pdf_to_two_pages(file_path, output_path)




# document_analysis_client = DocumentAnalysisClient(
#     endpoint=OCR_ENDPOINT,
#     credential=AzureKeyCredential(OCR_API_KEY)
# )
# 
# loader = DirectoryLoader(
#     '.',
#     glob="./truncated_*.pdf",
#     use_multithreading=True,
#     loader_cls=DocumentIntelligenceLoader,
#     loader_kwargs={"client": document_analysis_client, "model": "prebuilt-read"}
# )
# 
# 
# docs = loader.load()
# 
# # insert the documents in MongoDB Atlas with their embedding
# vector_search = MongoDBAtlasVectorSearch.from_documents(
#     documents=docs,
#     embedding=OpenAIEmbeddings(disallowed_special=()),
#     collection=MONGODB_COLLECTION,
#     index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
# )






# Perform a similarity search between the embedding of the query and the embeddings of the documents
# query = "Where does kingham youth soccer operate?"

# results = vector_search.similarity_search(query)

# OCR_ENDPOINT = os.environ["OCR_ENDPOINT"]
# OCR_API_KEY = os.environ["OCR_API_KEY"]
MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = "nonprofit-grader"
COLLECTION_NAME = "jan2022"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default"
client = MongoClient(MONGODB_URI)

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_URI,
    "nonprofit-grader" + "." + COLLECTION_NAME,
    OpenAIEmbeddings(disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)
qa_retriever = vector_search.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)


prompt_template = """Find organization(s) that deals with {question}, and tell me about them.


Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)



qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name='gpt-4'),
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

# query = "Find an organization that deals with soccer, and tell me about them."

# docs = qa({"query": query})
# 
# print(docs["result"])
# print(docs["source_documents"])
