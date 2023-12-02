import coloredlogs
import logging
import os
import json
import boto3
import traceback

from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import AmazonKendraRetriever
from langchain.llms.bedrock import Bedrock

coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level='INFO')
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID", None)
REGION = os.getenv('AWS_REGION', 'us-west-2')


def get_bedrock_client():
    bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
    return bedrock_client

def create_bedrock_llm(bedrock_client, model_version_id, model_args):
    bedrock_llm = Bedrock(
        model_id=model_version_id, 
        client=bedrock_client,
        model_kwargs=model_args,
        verbose=True, 
        )
    return bedrock_llm

def create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client
    

def lambda_handler(event, context):
    logging.info(f"Event is: {event}")
    # Parse event body
    event_body = json.loads(event["body"])
    question = event_body["query"]
    logging.info(f"Query is: {question}")
    
    
    try:

        #bedrock_embedding_model_id = args.bedrock_embedding_model_id

        bedrock_model_id = event_body["model_id"]
        temperature = event_body["temperature"]
        maxTokens = event_body["max_tokens"]
        logging.info(f"selected model id: {bedrock_model_id}")
        
        model_args = {}
        
        PROMPT_TEMPLATE = 'prompt-engineering/claude-prompt-template.txt'
        if bedrock_model_id == 'anthropic.claude-v2':
            model_args = {
                "max_tokens_to_sample": int(maxTokens),
                "temperature": float(temperature),
                "top_k": 250,
                "top_p": 0.1
            }
            PROMPT_TEMPLATE = 'prompt-engineering/claude-prompt-template.txt'
        elif bedrock_model_id == 'amazon.titan-text-express-v1':
            model_args = {
                "maxTokenCount": int(maxTokens),
                "temperature": float(temperature),
                "topP":1
            }
            PROMPT_TEMPLATE = 'prompt-engineering/titan-prompt-template.txt'
        elif bedrock_model_id == 'ai21.j2-mid-v1' or bedrock_model_id == 'ai21.j2-ultra-v1':
            model_args = {
                "maxTokens": int(maxTokens),
                "temperature": float(temperature),
                "topP":1
            }
            PROMPT_TEMPLATE = 'prompt-engineering/jurassic2-prompt-template.txt'
        else:
            model_args = {
                "max_tokens_to_sample": int(maxTokens),
                "temperature": float(temperature)
            }
            PROMPT_TEMPLATE = os.environ["PROMPT_TEMPLATE_CLAUDE"]

        # Read the prompt template from S3 bucket
        s3 = boto3.resource('s3')
        obj = s3.Object(S3_BUCKET_NAME, PROMPT_TEMPLATE) 
        prompt_template = obj.get()['Body'].read().decode('utf-8')
        logging.info(f"prompt: {prompt_template}")

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        #Create bedrock llm object
        bedrock_client = get_bedrock_client()
        bedrock_llm = create_bedrock_llm(bedrock_client, bedrock_model_id, model_args)
        #bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id)
        
        logging.info(f"Starting the chain with KNN similarity using Amazon Kendra, Bedrock FM {bedrock_model_id}")
        qa = RetrievalQA.from_chain_type(llm=bedrock_llm, 
                                    chain_type="stuff", 
                                    retriever=AmazonKendraRetriever(index_id=KENDRA_INDEX_ID),
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": PROMPT, "verbose": True},
                                    verbose=True)
        
        response = qa(question, return_only_outputs=False)
        
        source_documents = response.get('source_documents')
        source_docs = []
        for d in source_documents:
            source_docs.append(d.metadata['source'])
        
        output = {"answer": response.get('result'), "source_documents": source_docs}

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
            "body": json.dumps(output)
        }

        logging.info('new bedrock version tested successfully')
    except Exception as e:
        print('Error: ' + str(e))
        traceback.print_exc()

        # Handle exceptions and provide an error response
        error_message = str(e)
        error_response = {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
            "body": json.dumps({"error": error_message})
        }
        return error_response

