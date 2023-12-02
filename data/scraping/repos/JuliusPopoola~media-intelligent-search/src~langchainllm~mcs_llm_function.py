from langchain.retrievers import AmazonKendraRetriever
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.llms.bedrock import Bedrock

from datetime import datetime
import json
import boto3

def lambda_handler(event, context):
    
    statusCode = 200
    body = {}

    try:
        print(event)
    except Exception as exception:
        print(exception)
        statusCode = 500
        body = {
            'error': str(exception)
        }
        raise

    finally:
        print('Status Code:' + str(statusCode))
        print(json.dumps(body))
        return {
            'statusCode': statusCode,
            'body': json.dumps(body),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }   