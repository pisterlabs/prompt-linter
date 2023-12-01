from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock

import boto3
import json
import os
import sys
from utils import bedrock, print_ww

def get_titan_llm():
    bedrock_region_name='us-west-2'
    bedrock_endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com'
    aws_profile=None
    
    session = boto3.Session(profile_name=aws_profile)
    bedrock_client = session.client(service_name='bedrock-runtime',
                           region_name=bedrock_region_name,
                           endpoint_url=bedrock_endpoint_url)

    titan_parameteres  = {'maxTokenCount':3072, 
                         "temperature":0.01,
                         #"top_k":250,
                          #"top_p":1,
                          #"stop_sequences": ["\n\nHuman"]
                         }

    titan_llm = Bedrock(model_id = "amazon.titan-text-express-v1",
                        client = bedrock_client, 
                        model_kwargs = titan_parameteres 
                        )
    return titan_llm

def get_ai21_llm():
    
    bedrock_region_name='us-west-2'
    bedrock_endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com'
    aws_profile="jaybedrock"
    
    
    session = boto3.Session(profile_name=aws_profile)
    bedrock_client = session.client(service_name='bedrock-runtime',
                           region_name=bedrock_region_name,
                           endpoint_url=bedrock_endpoint_url)

    ai21_parameteres  = {'maxTokens':300, 
                         "temperature":0,
                         }

    ai21_llm = Bedrock(model_id = "ai21.j2-ultra",
                        client = bedrock_client, 
                        model_kwargs = ai21_parameteres 
                        )
    return ai21_llm

def get_claudeV2_llm():
    
    bedrock_region_name='us-west-2'
    bedrock_endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com'
    aws_profile= None #"jaybedrock"

    session = boto3.Session(profile_name=aws_profile)
    bedrock_client = session.client(service_name='bedrock-runtime',
                           region_name=bedrock_region_name,
                           endpoint_url=bedrock_endpoint_url)

    cluade_parameters = {
                   'max_tokens_to_sample':4096, 
                    "temperature":0,
                    "top_k":250,
                    "top_p":1,
                    "stop_sequences": ["\n\nHuman"],
                }

    claude_llm = Bedrock(model_id = 'anthropic.claude-v2',
                    client = bedrock_client, 
                    model_kwargs = cluade_parameters 
                    )
    return claude_llm

def get_claudeInstant_llm():
    bedrock_region_name='us-west-2'
    bedrock_endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com'
    aws_profile= None #"jaybedrock"

    session = boto3.Session(profile_name=aws_profile)
    bedrock_client = session.client(service_name='bedrock-runtime',
                           region_name=bedrock_region_name,
                           endpoint_url=bedrock_endpoint_url)

    cluade_parameters = {
                   'max_tokens_to_sample':4096, 
                    "temperature":0,
                    "top_k":250,
                    "top_p":1,
                    "stop_sequences": ["\n\nHuman"],
                }

    claude_llm = Bedrock(model_id = 'anthropic.claude-instant-v1',
                    client = bedrock_client, 
                    model_kwargs = cluade_parameters 
                    )
    return claude_llm

def get_claudeV1_llm():
    bedrock_region_name='us-west-2'
    bedrock_endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com'
    aws_profile= None #"jaybedrock"

    session = boto3.Session(profile_name=aws_profile)
    bedrock_client = session.client(service_name='bedrock-runtime',
                           region_name=bedrock_region_name,
                           endpoint_url=bedrock_endpoint_url)

    cluade_parameters = {
                   'max_tokens_to_sample':4096, 
                    "temperature":0,
                    "top_k":250,
                    "top_p":1,
                    "stop_sequences": ["\n\nHuman"],
                }

    claude_llm = Bedrock(model_id = 'anthropic.claude-v2',
                    client = bedrock_client, 
                    model_kwargs = cluade_parameters 
                    )
    return claude_llm


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MAX_HISTORY_LENGTH = 5

def build_chain(model="claudeV2"):
    region = os.environ["AWS_REGION"]
    kendra_index_id = os.environ["KENDRA_INDEX_ID"]

    if model=="claudeV2":
        llm=get_claudeV2_llm()
    elif model=="claudeV1":
        llm=get_claudeV1_llm()
    elif model=="claudeInstant":
        llm=get_claudeInstant_llm()
    elif model=="titan":
        llm=get_titan_llm()
    elif model=="ai21":
        llm=get_ai21_llm()
    

    retriever = AmazonKendraRetriever(index_id=kendra_index_id,region_name=region)
    prompt_template = """\n\nHuman:The following is a friendly conversation between a human and an AI. \
        The AI is talkative and provides lots of specific details from its <context>. If the AI does not \
        know the answer to a <question>, it truthfully says it does not know.
        <context>{context}</context>
        Answer the below question as truthfully as possible based on above <context>.
        <question>{question}</question>
        \n\nAssistant:
        """

    PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    return qa

def run_chain(chain, prompt: str, history=[]):
    result = chain(prompt)
    # To make it compatible with chat samples
    return {
        "answer": result['result'],
        "source_documents": result['source_documents']
    }

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        chain = build_chain(sys.argv[1])
    else:
        chain = build_chain()

    result = run_chain(chain, "What is Sagemaker?")
    print(result['answer'])
    if 'source_documents' in result:
        print('Sources:')
        for d in result['source_documents']:
          print(d.metadata['source'])
