import os
import openai
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import lark

openai_api_key = os.environ.get('OPENAI_API_KEY')
print(len(openai_api_key))

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(vectordb._collection.count())

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="test 1111 ",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="test 222",
        type="integer",
    ),
]

document_content_description = "Lecture notes"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)

question = 'what is the estimated length date of processing?'

docs = retriever.get_relevant_documents(question)

for d in docs:
    print(d.metadata)