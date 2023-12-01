
'''import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
'''

from langchain.embeddings import OpenAIEmbeddings

config = {
    "splitter_type": "CharacterTextSplitter",
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "length_function" : len, 
    "separators" : ["}"],  #[" ", ",", "\n"]
    "embedding": OpenAIEmbeddings(), #  includes a pull of the open api key
    "embedding_dims": 1536,
    "search_type": "mmr",
    "k": 4,
    'fetch_k': 10,   # fetch 30 docs then select 4
    'lambda_mult': .7,    # 0= max diversity, 1 is min. default is 0.5
    "score_threshold": 0.5,
    "model": "gpt-3.5-turbo-16k",
    "temperature": 0.7,
    "chain_type": "stuff",
}

#CONFIG: qdrant
qdrant_collection_name = "ASK_vectorstore"
qdrant_path = "/tmp/local_qdrant" # Only required for local instance /private/tmp/local_qdrant



    #-----------------------------------
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import tiktoken
import pickle

llm=ChatOpenAI(model=config["model"], temperature=config["temperature"]) #keep outside the function so it's accessible elsewhere in this notebook

query = []



def qdrant_connect_local():
    print("attempting to assign client")
    
    if 'client' in globals():
        return globals()['client']  # Return the existing client
    client = QdrantClient(path=qdrant_path)  # Only required for a local instance
    return client



def qdrant_connect_cloud(api_key):
    print("attempting to assign client")
    
    if 'client' in globals():
        return globals()['client']  # Return the existing client
    client = QdrantClient(
    "https://0c82e035-1105-40f2-a0bd-ecc44a016f15.us-east4-0.gcp.cloud.qdrant.io", 
    prefer_grpc=True,
    api_key=api_key,
    )
    return client



def create_langchain_qdrant(client):
    '''create a langchain vectorstore object'''
    qdrant = Qdrant(
        client=client, 
        collection_name=qdrant_collection_name, 
        embeddings=config["embedding"]
    )
    return qdrant

    

def init_retriever_and_generator(qdrant):
    '''initialize a document retriever and response generator'''
    retriever = qdrant.as_retriever(
        search_type=config["search_type"], 
        search_kwargs={'k': config["k"], "fetch_k": config["fetch_k"], "lambda_mult": config["lambda_mult"], "filter": None}, # filter documents by metadata
    )
    return retriever


system_message_prompt_template = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=['context'],
        template="Use the following pieces of context to answer the users question. Be sure to include all the requirements and tasks in your response. If the question is about qualification, certification or currency, then follow these steps: 1.Determine the name of the qualifiction or certification. 2. Determine whether the question is about initial qualificaiton or currency maintenance. Each have different requirements. 3. Determine what program the qualification or certification belongs, such as Boat Crew program or Aviation program. 4. Determine any requirements that apply to all positions and certifications in that program as well as the specific requirements for the certification. For example, a Coxswain is a certification in the boat crew program. The Boat Crew program has requirements such as annual surface operations workshop. Additionally, coxswain has the requirement to complete a navigation test. Likewise, A Co-Pilot is a certification in the Aviation program. The Aviation program has requirements for all flight crewmembers that apply to Co-Pilot and First Pilot. First Pilot and Co-Pilot are Pilot flight crew positions, so they have Pilot requirements apply to First Pilot and Co-Pilot. Co-Pilot and First Pilot may have additional requirements specific to their certification.  \nIf you don't know the answer, just say I don't know, don't try to make up an answer.\n----------------\n{context}"
    )
)


def rag(query, retriever):
    '''run a RAG completion'''

    llm_chain = LLMChain(
        prompt=ChatPromptTemplate(input_variables=['context', 'question'], messages=[system_message_prompt_template, HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question'], template='{question}'))]),
        llm=llm,
        )

    rag_instance = RetrievalQA(
        combine_documents_chain=StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name='context'),
        return_source_documents=True,
        retriever=retriever
    )
    response = rag_instance({"query": query})
    return response


def rag_old1(query, retriever):
    '''run a RAG completion'''

    rag_instance = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=config["chain_type"],
        retriever=retriever,
        return_source_documents=True,
    )
    response = rag_instance({"query": query})
    return response



def rag_dummy(query, retriever):
    '''returns a dummy canned response'''

    with open("test/dummy_response.pkl", "rb") as file:
        dummy_response = pickle.load(file)
    return dummy_response
        


def create_short_source_list(response):
    '''Extracts a list of sources with no description 
    
    The dictionary has three elements (query, response, and source_documents). 
    Inside the third is a list with a custom object Document 
    associated with the key 'source_documents'
    '''

    markdown_list = []
    
    for i, doc in enumerate(response['source_documents'], start=1):
        page_content = doc.page_content  
        source = doc.metadata['source']  
        short_source = source.split('/')[-1].split('.')[0]  
        page = doc.metadata['page']  
        markdown_list.append(f"*{short_source}*, page {page}\n")
    
    short_source_list = '\n'.join(markdown_list)
    return short_source_list



def create_long_source_list(response):
    '''Extracts a list of sources along with full source
    
    response is a dictionary with three keys:
    dict_keys(['query', 'result', 'source_documents'])
    'source_documents' is a list with a custom object Document 
    '''
    
    markdown_list = []
    
    for i, doc in enumerate(response['source_documents'], start=1):
        page_content = doc.page_content  
        source = doc.metadata['source']  
        short_source = source.split('/')[-1].split('.')[0]  
        page = doc.metadata['page']  
        markdown_list.append(f"**Reference {i}:**    *{short_source}*, page {page}   {page_content}\n")
    
    long_source_list = '\n'.join(markdown_list)
    return long_source_list



def count_tokens(response):
    ''' counts the tokens from the response'''
    encoding = tiktoken.encoding_for_model(config["model"])
    query_tokens = encoding.encode(response['query'])
    query_length = len(query_tokens)
    source_tokens = encoding.encode(str(response['source_documents']))
    source_length = len(source_tokens)
    result_tokens = encoding.encode(response['result'])
    result_length = len(result_tokens)
    tokens = encoding.encode(str(response))
    tot_tokens = len(tokens)
    return query_length, source_length, result_length, tot_tokens


import requests


def get_openai_api_status():
    components_url = 'https://status.openai.com/api/v2/components.json'
    status_message = ''

    try:
        response = requests.get(components_url)
        # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

        # Parse the JSON response
        components_info = response.json()
        components = components_info.get('components', [])

        # Find the component that represents the API
        api_component = next(
            (component for component in components if component.get('name', '').lower() == 'api'), None)

        if api_component:
            # Set the status message to the status of the API component
            status_message = api_component.get('status', '')
        else:
            status_message = 'API component not found'

    except requests.exceptions.HTTPError as http_err:
        status_message = f'HTTP error occurred: {repr(http_err)}'
    except Exception as err:
        status_message = f'Other error occurred: {repr(err)}'

    return status_message



# Example usage in another script
if __name__ == "__main__":
   
   
    # Replace 'your_query' with the actual query you want to pass to rag
    query = 'your_query'
    response = rag(query, retriever) #thisn is slightly different from the notebook
    
    # Call other functions to process the response
    short_source_list = create_short_source_list(response)
    long_source_list = create_long_source_list(response)
    source_length, source_tokens, tot_tokens = count_tokens(response)
