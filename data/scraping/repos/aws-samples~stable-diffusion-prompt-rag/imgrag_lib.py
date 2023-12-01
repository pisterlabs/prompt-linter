import os
import json, boto3, base64
from io import BytesIO

from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain import LLMChain

session = boto3.Session(
#    profile_name=os.environ.get("BWB_PROFILE_NAME")
) #sets the profile name to use for AWS credentials

bedrock_client = session.client(
    service_name='bedrock-runtime', #creates a Bedrock client
    region_name=os.environ.get("BWB_REGION_NAME"),
    endpoint_url=os.environ.get("BWB_ENDPOINT_URL")
) 

###############################################################################
######################### Part 1: LLM Model Stetup #############################

# We will be using the Claude LLM to generate texts and the Titan Embedding Model to generate Embeddings  
# Select models and parameters
bedrock_llm_id = "anthropic.claude-v2"
bedrock_embed_id = "amazon.titan-embed-text-v1"
bedrock_model_id = "stability.stable-diffusion-xl-v0" #use the Stable Diffusion model

def get_llm():
    
    model_kwargs =  { 
        "max_tokens_to_sample": 1024, 
        "temperature": 1, 
        "top_k": 250,
        "top_p": 1,
        #"stop_sequences": ["\n\nHuman:"]
    }
    
    llm = Bedrock(
        model_id=bedrock_llm_id, #set the foundation model
        client=bedrock_client, 
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm


def get_index(): #creates and returns an in-memory vector store to be used in the application
    
    embeddings = BedrockEmbeddings(
        model_id=bedrock_embed_id, #set the foundation model
        client=bedrock_client, 
    ) #create a Titan Embeddings client

    loader = CSVLoader(file_path="./prompts_unique.csv")    
    documents = loader.load()
    
    index_from_loader = FAISS.from_documents(documents, embeddings)
        
    return index_from_loader #return the index to be cached by the client app

def sementic_search(index, original_prompt): #rag client function
        
    relevant_prompts = index.similarity_search(original_prompt)    

    list_prompts = []
    for i in range(len(relevant_prompts)):
        list_prompts.append(relevant_prompts[i].page_content)
    
    return list_prompts


def get_rag_response(original_prompt, selected_prompt): #rag client function
    
    llm = get_llm()

    # Create a Prompt Template
    prompt_template = """This app is to generate prompt for image generation. the user will provide Original Prompt for image generation. Based on Selected prompt, Only slightly revise Original Prompt. \
                    Please keep the Generated Prompt clear, complete, and less than 50 words. 
                    Original Prompt: {original_prompt}\n\n
                    Selected Prompt: {selected_prompt}\n\n
                    Generated Prompt: """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["original_prompt", "selected_prompt"])

    llm_chain = LLMChain(
         llm=llm,
         prompt=PROMPT
     )

    result = llm_chain.run({"original_prompt": original_prompt, "selected_prompt": selected_prompt,})
    
    return result


###############################################################################
######################### Part 2: SD Model Stetup #############################

def get_response_image_from_payload(response): #returns the image bytes from the model response payload

    payload = json.loads(response.get('body').read()) #load the response body into a json object
    images = payload.get('artifacts') #extract the image artifacts
    image_data = base64.b64decode(images[0].get('base64')) #decode image

    return BytesIO(image_data) #return a BytesIO object for client app consumption

def get_image_response(prompt_content): #text-to-text client function
    
    request_body = json.dumps({"text_prompts": 
                               [ {"text": prompt_content } ], #prompts to use
                               "cfg_scale": 9, #how closely the model tries to match the prompt
                               "steps": 50, }) #number of diffusion steps to perform
    
    response = bedrock_client.invoke_model(body=request_body, modelId=bedrock_model_id) #call the Bedrock endpoint
    
    output = get_response_image_from_payload(response) #convert the response payload to a BytesIO object for the client to consume
    
    return output