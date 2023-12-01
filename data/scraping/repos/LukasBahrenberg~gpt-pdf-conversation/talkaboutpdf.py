import os
import dotenv
import pinecone

# langchain imports
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


# load .env file
dotenv.load_dotenv()


# load and chunk up pdf
loader = UnstructuredPDFLoader("./oe-28-14-20489.pdf")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = loader.load_and_split(text_splitter)
print(f'You have {len(documents)} document(s) in your data.')
print(f'There are {len(documents[0].page_content)} characters in your document.')


# embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


# pinecone init
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index_name = api_key=os.getenv("INDEX_NAME"),

docsearch = Pinecone.from_documents(documents, embeddings, index_name=index_name)


# llms
llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY"))
chain = load_qa_chain(llm, chain_type="stuff")


while True:
    # define query and make according vector db search
    query = input("> ")
    docs = docsearch.similarity_search(query)

    # make llm api call
    for chunk in chain.run(input_documents=docs, question=query):
        if chunk is not None:
            print(chunk, end='')
        
    print('\n')