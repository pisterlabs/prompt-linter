import pinecone
import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
import os
import openai



# Set Pinecone API and Environment
pinecone_api_key=os.getenv('PINECONE_API_KEY') 
pinecone_environment=os.getenv('PINECONE_ENVIRONMENT') 
index_name=os.getenv('PINECONE_INDEX_NAME')

pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

index = pinecone.GRPCIndex(index_name)

openai.api_key = os.getenv("OPENAI_API_KEY")
model_name="gpt-3.5-turbo"
text_field = "text"

# completion llm
llm = ChatOpenAI(
    openai_api_key=openai.api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai.api_key
)

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)


qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

template = """
You are an expert in geopolitical analysis specialized in metageopolitics (explained in metageopolitical knowledge graph section) and are tasked with creating a compelling, well-structured, and informative weekly essay (in weekly scan section) for C-suite executives. The essay should cover the most critical priorities grouped into PESTLE (Political, Economic, Sociocultural, Technological, Legal, and Environmental) sections. Your goal is to provide concise yet comprehensive insights, allowing executives to make informed decisions for their organizations.

Context: {context}

Question: {question}

Answer: """

question = """
explain according to the knowledge of Russia invaded Ukraine in 2022, why Prigozhin fight with Putin, and what's next?"
"""

with open('web.txt', 'r') as file:
    context = file.read()

# Now you can use 'context' in the rest of your code


# Create a PromptTemplate object
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Create a new essay by formatting the context and query parameters with the PromptTemplate object
newessay = prompt_template.format(context=context, question=question)

# Initialize an OpenAI object
openai_instance = OpenAI(
    model_name="gpt-4",
    openai_api_key=openai.api_key
)

# Generate a response from the OpenAI API
response = openai_instance(newessay)

print(response)

response2 = qa_with_sources("Why Putin invaded Ukraine?")

print(response2)

