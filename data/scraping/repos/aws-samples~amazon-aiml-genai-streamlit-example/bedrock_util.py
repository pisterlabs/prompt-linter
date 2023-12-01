import boto3
import botocore
from botocore.config import Config
import json
import constants as const
import anthropic

region = const.region_name
url = const.bedrock_ep_url

config = Config(
   retries = {
      'max_attempts': 3,
      'mode': 'standard'
   }
)


def bedrock_test():
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock',region_name=region,endpoint_url=url, config=config)
    output_text = bedrock.list_foundation_models()
    return output_text

def call_titan(prompt):
    #create boto3 client for bedrock
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock',region_name=region,endpoint_url=url, config=config)

    body = json.dumps(
        {"inputText": prompt, 
         "textGenerationConfig": {
             "maxTokenCount": 4096,
             "temperature": 0.5,
             "topP": 1,
             # "stopSequences": []
         }
          })
    model_id = "amazon.titan-tg1-large"
    content_type = 'application/json'
    accept = 'application/json'

    response = bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)
    response_body = json.loads(response.get('body').read())

    return response_body['results'][0]['outputText']
    

#Claude V2 and paramters
def call_model_claude(prompt):
    #create boto3 client for bedrock
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock',region_name=region,endpoint_url=url, config=config)

    body = json.dumps(
        {"prompt": anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT, 
         "max_tokens_to_sample": 1024,
         "temperature":0.5,
         "top_p":1,
         "top_k":250,
         "stop_sequences":[anthropic.HUMAN_PROMPT]
          })
    # model_id = 'anthropic.claude-v2'
    model_id = 'anthropic.claude-instant-v1'
    content_type = 'application/json'
    accept = 'application/json'

    response = bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)
    response_body = json.loads(response.get('body').read())

    return response_body['completion']

def call_model_jurassic(prompt):
    #create boto3 client for bedrock
    session = boto3.Session()
    bedrock = session.client(service_name='bedrock',region_name=region,endpoint_url=url, config=config)

    body = json.dumps(
        {"prompt": prompt + f"\n", 
         "maxTokens": 1024,
         "temperature": 0.5,
         "topP": 1,
         "stopSequences": [],
         "countPenalty": {"scale": 0},
         "presencePenalty": {"scale": 0},
         "frequencyPenalty": {"scale": 0}
          })
    model_id = "ai21.j2-jumbo-instruct"
    content_type = 'application/json'
    accept = 'application/json'

    response = bedrock.invoke_model(body=body, modelId=model_id, accept=accept, contentType=content_type)
    response_body = json.loads(response.get('body').read())

    return response_body['completions'][0]['data']['text']

inference_modifier_titan = { 
    "maxTokenCount": 4096,
    "stopSequences": [],
    "temperature":0.1,  
    "topP":0.9
}

inference_modifier_claude = {
    'max_tokens_to_sample':4096,
    "temperature":0.5,
    "top_k":250,
    "top_p":1,
    "stop_sequences": ["\n\nHuman"]
}