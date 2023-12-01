from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from config import *

embeddings = OpenAIEmbeddings(
    openai_api_key=openai_key
)

# Pinecone Vector DB
pinecone.init(
    api_key=pinecone_key,
    environment=pinecone_env
)

# Store vectors
# to use existing vectors: docsearch = Pinecone.from_existing_index(embeddings, index_name=pinecone_index_name)
docsearch = Pinecone.from_existing_index(pinecone_index_name, embeddings, namespace=pinecone_namespace)

# Find similar docs based on prompt
userPrompt = input("Your question: ")
docs = docsearch.similarity_search(userPrompt, 4)

# Inject document into prompt
promptTemplate = """
You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say you don't know. DO NOT try to make up an answer. 
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
\n\n {promptDocuments}
\n\n Question: {userPrompt}
\n\n Helpful answer in markdown:
"""

promptDocuments = ""
for k,v in enumerate(docs):
    promptDocuments += v.page_content

# Ask GPT-3.5-turbo
chat = ChatOpenAI(
    openai_api_key=openai_key,
    model_name="gpt-3.5-turbo"
)

prompt = PromptTemplate(
    input_variables=["userPrompt", "promptDocuments"],
    template=promptTemplate
)

chatResult = chat.generate([[
    SystemMessage(content="You are a helpful assistant who uses document from a knowledge base to answer questions."),
    HumanMessage(content=prompt.format(userPrompt=userPrompt, promptDocuments=promptDocuments))
]])

print(chatResult.llm_output)
print("Answer: "+ chatResult.generations[0][0].text)

print("Sources:")
for k,v in enumerate(docs):
    print("\tDocument "+ str(k))
    print("\t\t"+ v.page_content.replace("\n\n", ""))
