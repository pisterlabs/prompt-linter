from dotenv import load_dotenv

load_dotenv('E:\\MyOnlineCourses\\ML_Projects\\dagshubrepos\\pokmn\\tokens.env')

##########################################################################
# ZERO TO HERO - Deeplake and langchain course
##########################################################################

# SECTION [1]
# DEEP LAKE VectorStore

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType


# instantiate the LLM and embeddings models
llm = OpenAI(model="text-davinci-003", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

texts_1 = [
            "Napoleon Bonaparte was born in 15 August 1769",
            "Louis XIV was born in 5 September 1638"
        ]


def create_deeplake_dataset(texts, chunk_size=1000, chunk_overlap=0, embeddings_model="text-embedding-ada-002"):
    """
    Create a DeepLake dataset from a list of texts.
    
    Args:
        texts (list): List of texts to create the dataset from.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Number of characters to overlap between chunks.
        embeddings_model (str): Model name for embeddings.
        
    Returns:
        None
    """
    
    # Create embeddings object
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    
    # Split texts into documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents(texts)

    # Create Deep Lake dataset
    my_activeloop_org_id = "mohdnassgabr" 
    my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

    # Add documents to the Deep Lake dataset with error handling
    try:
        db.add_documents(docs)
        return db
    except Exception as e:
        print(f"Error adding documents to the dataset: {e}")
        # Handle the error appropriately
    
    
def create_retrieval_qa(db):
    """
    Initializes a RetrievalQA object using the specified database.

    Args:
        db: The database object used for retrieval.

    Returns:
        None

    Raises:
        None
    """
    try:
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever()
        )
        return retrieval_qa
    except Exception as e:
        # Handle the exception here
        print(f"An error occurred: {e}")
        # Optionally, raise or propagate the exception further

dl_db = create_deeplake_dataset(texts=texts_1)
retrieval_qa = create_retrieval_qa(db=dl_db)

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
        ),
    ]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True)

def agent(query):   
    response = agent.run(query)
    print(response)

agent(query="When was Napoleone born?")