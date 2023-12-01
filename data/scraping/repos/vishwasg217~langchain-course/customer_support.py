from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate
from dotenv import dotenv_values

config = dotenv_values(".env")
OPEN_AI_API = config["OPEN_AI_API"]
ACTIVELOOP_TOKEN = config["ACTIVELOOP_TOKEN"]

urls = [
    'https://beebom.com/what-is-nft-explained/',
    'https://beebom.com/how-delete-spotify-account/',
    'https://beebom.com/how-download-gif-twitter/',
    'https://beebom.com/how-use-chatgpt-linux-terminal/'
]


# loading the data from articles 
loader = SeleniumURLLoader(urls)
docs = loader.load()
print('Loaded {} documents'.format(len(docs)))
# splitting the documents into chunks of 1000 characters
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splitted_docs = splitter.split_documents(docs)
print('Splitted {} documents'.format(len(splitted_docs)))
# initializing the Open AI embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPEN_AI_API)


# activeloop credentials for creating a new database
my_activeloop_org_id = "vishwasg217"
my_activeloop_dataset_name = "langchain_course_customer_support"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
print('Accessed database at {}'.format(dataset_path))
db.add_documents(splitted_docs)

query = "Explain linear regression in brief"
docs = db.similarity_search(query)
retreived_chunks = [doc.page_content for doc in docs]
formatted_chunks = "\n\n".join(retreived_chunks)

# for doc in docs:
#     print(doc.page_content)
#     print('----------------------')

template = """
You are a customer chatbot that answers questions using the following information:

{formatted_chunks}

Use only the above information to answer the questions below. If you don't know the answer, respond with "I don't know".

Question: {query}

Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["formatted_chunks", "query"])
formatted_prompt = prompt.format(formatted_chunks=formatted_chunks, query=query)

llm = OpenAI(model="text-davinci-003", temperature=0.0, openai_api_key=OPEN_AI_API)
answer = llm(formatted_prompt)
print(answer)
