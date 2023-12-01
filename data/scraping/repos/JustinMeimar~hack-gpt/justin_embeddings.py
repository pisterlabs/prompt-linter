# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate

# llm = OpenAI(temperature=0.9)

# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What is a good name for a company that makes {product}?",
# )
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory

# conversation = ConversationChain(
#     llm=chat,
#     memory=ConversationBufferMemory()
# )

# conversation.run("Answer briefly. What are the first 3 colors of a rainbow?")
# # -> The first three colors of a rainbow are red, orange, and yellow.
# conversation.run("And the next 4?")
# -> The next four colors of a rainbow are green, blue, indigo, and violet.

import os
import pinecone

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

def query_retriever(vectorstore):
    
    metadata_field_info = [
        AttributeInfo(
            name="genre",
            description="The genre of the movie",
            type="string or list[string]",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(
            name="rating", description="A 1-10 rating for the movie", type="float"
        ),
    ]

    document_content_description = "Brief summary of a movie"
    llm = OpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm, vectorstore, document_content_description, metadata_field_info, verbose=True
    )

    while(True):
        query = input("Enter a query to search in the vectorstore: ")
        response = retriever.get_relevant_documents(query)
        print(response)

def extract_metadata_from_json(listing_json):
    return 

def get_doccuments():
    docs = [
        Document(
            page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
            metadata={"year": 1993, "rating": 7.7, "genre": ["action", "science fiction"]},
        ),
        Document(
            page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
            metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
        ),
        Document(
            page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
            metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
        ),
        Document(
            page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
            metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
        ),
        Document(
            page_content="Toys come alive and have a blast doing so",
            metadata={"year": 1995, "genre": "animated"},
        ),
        Document(
            page_content="Three men walk into the Zone, three men walk out of the Zone",
            metadata={
                "year": 1979,
                "rating": 9.9,
                "director": "Andrei Tarkovsky",
                "genre": ["science fiction", "thriller"],
                "rating": 9.9,
            },
        ),
    ]
    return docs

def create_index():
    idxs = pinecone.list_indexes()
    if idxs == []:
        pinecone.create_index("langchain-self-retriever-demo", dimension=1536)
    return 

def init_everything():
    load_dotenv()

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"]
    )

    create_index()
    print("== Created index")

    docs = get_doccuments()
    print("== Assembled Documents to embedd")

    embeddings = OpenAIEmbeddings()
    print("== Got OpenAI embeddings")
    
    vectorstore = Pinecone.from_documents(
        docs, embeddings, index_name="langchain-self-retriever-demo"
    )
    print("== Upserted embedded vectors")

    query_retriever(vectorstore=vectorstore)

def get_mock_listings():

    return 

if __name__ == "__main__":
    init_everything()