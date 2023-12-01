import os
from dotenv import load_dotenv

# Load environment variables
if load_dotenv("../.env"):
    print("Found OpenAPI Base Endpoint: " + os.getenv("OPENAI_API_BASE"))
else: 
    print("No file .env found")

openai_api_type = os.getenv("OPENAI_API_TYPE")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
completion_model = os.getenv("OPENAI_COMPLETION_MODEL")
embedding_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='movies.csv', source_column='original_title', encoding='utf-8', csv_args={'delimiter':',', 'fieldnames': ['id', 'original_language', 'original_title', 'popularity', 'release_date', 'vote_average', 'vote_count', 'genre', 'overview', 'revenue', 'runtime', 'tagline']})
data = loader.load()
data = data[1:11] # reduce dataset if you want
print('Loaded %s movies' % len(data))

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI

# Create an Embeddings Instance of Azure OpenAI
embeddings = OpenAIEmbeddings(
    deployment=embedding_name,
    chunk_size=1
) 

# Create a Chat Completion Instance of Azure OpenAI
llm = AzureChatOpenAI(
    openai_api_type = openai_api_type,
    openai_api_version = openai_api_version,
    openai_api_base = openai_api_base,
    openai_api_key = openai_api_key,
    deployment_name = deployment_name,
    model_name=completion_model
)

from langchain.vectorstores import Qdrant

url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    data,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="my_movies",
)

vectorstore = qdrant

query = "What is the best 80s movie I should look at?"
found_docs = vectorstore.similarity_search(query)

print(found_docs[0].metadata['source'])

retriever = vectorstore.as_retriever(search_type="mmr")

query = "Which movies are about space travel?"
print(retriever.get_relevant_documents(query)[0].metadata['source'])

from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

index_creator = VectorstoreIndexCreator(embedding=embeddings)
docsearch = index_creator.from_loaders([loader])

llm = AzureChatOpenAI(
    openai_api_type = openai_api_type,
    openai_api_version = openai_api_version,
    openai_api_base = openai_api_base,
    openai_api_key = openai_api_key,
    deployment_name = deployment_name,
    model_name = completion_model
)
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question", return_source_documents=True)

query = "Do you have a column called popularity?"
print ("First query: " + query)
response = chain({"question": query})
print(response['result'])
print ("Source documents for first query")
print(response['source_documents'])
print ("\n")

query = "What is the movie with the highest popularity?"
print ("Second query: " + query)
response = chain({"question": query})
print(response['result'])
print("Source Documents for second query")
print(response['source_documents'])