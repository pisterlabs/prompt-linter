from langchain.llms import OpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
import os

import pinecone


pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_API_ENV"])

embeddings = OpenAIEmbeddings()
# from langchain.llms import OpenAI
metadata_field_info = [
    AttributeInfo(name="person_name",
                  description="The name of the person", type="string"),
    AttributeInfo(
        name="topic", description="The topic of the document", type="string"),
    AttributeInfo(
        name="text", description="The Content of the document", type="string"),
    AttributeInfo(name="chunk",
                  description="The number of the document chunk", type="int"),
    AttributeInfo(
        name="url", description="The name of url or document", type="string"),
    AttributeInfo(
        name="source", description="The source of the document", type="string"),
]

# Define document content description


document_content_description = "Content of the document"

vectorstore = Pinecone.from_existing_index(
    "embedings1", embeddings, "text", namespace="positive")
# Initialize OpenAI embeddings and LLM
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)

result = retriever.get_relevant_documents(input("sta trazimo? "))
print(result)


