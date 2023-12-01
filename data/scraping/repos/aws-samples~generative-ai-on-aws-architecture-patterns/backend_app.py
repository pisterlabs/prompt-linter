import boto3
import json
import os

from langchain.chains import ConversationChain
from langchain.llms import SagemakerEndpoint
from langchain.prompts import PromptTemplate
from langchain.llms.sagemaker_endpoint import ContentHandlerBase, LLMContentHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from llm.sagemaker_async_endpoint import *

REGION = os.environ.get('AWS_REGION', os.environ.get('AWS_DEFAULT_REGION'))
SM_ENDPOINT_NAME = os.environ.get('SM_ENDPOINT_NAME')
LLM_CONTEXT_LENGTH = os.environ.get('LLM_CONTEXT_LENGTH', '2048')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-instant-v1')
CHAT_WINDOW_SIZE = os.environ.get('CHAT_WINDOW_SIZE', '5')
LLM_MEMORY_TABLE = os.environ.get('LLM_MEMORY_TABLE', 'LLMChatMemoryTable')

# Content Handler for Falcon40b-instruct
class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs):
        input_str = json.dumps(
            {
                "inputs": prompt,
                "parameters":
                {
                    "do_sample": False,
                    "repetition_penalty": 1.1,
                    "return_full_text": False,
                    "max_new_tokens": 1024
                }
            }
        )
        return input_str.encode('utf-8')

    def transform_output(self, output):
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]

content_handler = ContentHandler()

def get_llm(use_bedrock):
    if use_bedrock:
        llm = Bedrock(client=boto3.client(service_name='bedrock-runtime'),
                    model_id=BEDROCK_MODEL_ID
        )
    else:
        # SageMaker langchain integration, to assist invoking SageMaker endpoint.
        llm = SagemakerEndpoint(
            endpoint_name=SM_ENDPOINT_NAME,
            region_name=REGION,
            # model_kwargs={}
            content_handler=content_handler,
        )
    return llm

def lambda_handler(event, context):

    try:
        print(event)

        body = json.loads(event['body'])
        print(body)

        query = body['query']
        uuid = body['uuid']
        use_bedrock = body.get('USE_BEDROCK')
        print(query)
        print(uuid)
        print(use_bedrock)

        llm = get_llm(use_bedrock)

        message_history = DynamoDBChatMessageHistory(
            table_name=LLM_MEMORY_TABLE,
            session_id=uuid
        )

        memory = ConversationBufferWindowMemory(
            chat_memory=message_history,
            return_messages=True,
            k=int(CHAT_WINDOW_SIZE),
        )

        chain = ConversationChain(
            llm=llm,
            verbose=True,
            memory=memory,
        )

        response = chain.run(query)
        clean_response = response.replace('\n','').strip()
        status_code = 200
        print(clean_response)
    except Exception as e:
        print(e)
        status_code = 500
        clean_response = e
    return {
        'statusCode': status_code,
        'body': json.dumps(f'{clean_response}')
    }
