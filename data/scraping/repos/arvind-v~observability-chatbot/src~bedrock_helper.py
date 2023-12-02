from langchain.schema import HumanMessage
from langchain.chat_models import BedrockChat
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from opensearch_helper import OpenSearchHelper
from param_store_helper import ParameterStoreHelper
import boto3
import json
import streamlit as st

class BedrockHelper:
    
    def __init__ (self, question = "What is Amazon Bedrock?"):
        self.question = question
    
    ps = ParameterStoreHelper('/eksworkshop/eks-workshop-1026/opensearch')
    opensearch = OpenSearchHelper (host = ps.host,
                               user = ps.user,
                               password = ps.password)
    
    #This uses Berock APIs without Langchain
    def invokeLLM (self, logs):
        # Setup Bedrock client
        bedrock = boto3.client('bedrock-runtime', 'us-west-2', endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com')
        # configure model specifics such as specific model
        modelId = 'anthropic.claude-v2'
        accept = 'application/json'
        contentType = 'application/json'
        # prompt that is passed into the LLM with the Kendra Retrieval context and question
        # TODO: FEEL FREE TO EDIT THIS PROMPT TO CATER TO YOUR USE CASE
        prompt_data = f"""\n\nHuman:    
    Answer the following question to the best of your ability based on the context provided.
    Provide an answer and provide sources and the source link to where the relevant information can be found. Include this at the end of the response
    Do not include information that is not relevant to the question.
    Only provide information based on the context provided, and do not make assumptions
    Only Provide the source if relevant information came from that source in your answer
    Use the provided examples as reference
    ###
    Question: {self.question}
    
    Context: {logs}
    
    ###
    
    \n\nAssistant:
    
    """
        # body of data with parameters that is passed into the bedrock invoke model request
        # TODO: TUNE THESE PARAMETERS AS YOU SEE FIT
        body = json.dumps({"prompt": prompt_data,
                           "max_tokens_to_sample": 8191,
                           "temperature": 0,
                           "top_k": 250,
                           "top_p": 0.5,
                           "stop_sequences": []
                           })
        # Invoking the bedrock model with your specifications
        response = bedrock.invoke_model(body=body,
                                        modelId=modelId,
                                        accept=accept,
                                        contentType=contentType)
        # the body of the response that was generated
        response_body = json.loads(response.get('body').read())
        # retrieving the specific completion field, where you answer will be
        answer = response_body.get('completion')
        # returning the answer as a final result, which ultimately gets returned to the end user
        return answer
        
        
    # Retrieve last nn minutes of documents from the 'eks-pod-logs' OpenSearch index
    def get_logs (self, client, index='eks-pod-logs', minutes=15):
        documents = self.opensearch.get_documents(client, index, minutes)
        logs = []
        for document in documents:
            log_entry = {}
            log_entry["@timestamp"] = document["_source"]["@timestamp"]
            log_entry["log_stream"] = document["_source"]["stream"]
            log_entry["pod_name"] = document["_source"]["kubernetes"]["pod_name"]
            log_entry["namespace"] = document["_source"]["kubernetes"]["namespace_name"]        
            log_entry["pod_labels"] = document["_source"]["kubernetes"]["labels"]        
            log_entry["container_name"] = document["_source"]["kubernetes"]["pod_name"]        
            log_entry["host"] = document["_source"]["kubernetes"]["host"]
            log_entry["log_message"] = document["_source"]["log"]
    
            logs.append(log_entry)
        return logs
        
    def invokeBedrockChat(self): 
      try: 
        chat = BedrockChat(
            model_id="anthropic.claude-v2",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            model_kwargs={
              #"maxTokenCount": 8191,
              #"stopSequences": [],
              #"topK": 250,
              #"topP": 0.5,
              "temperature": 0.0
              }
        )
      
        messages = [
            HumanMessage(
                content= self.question
            )
        ]
        answer = chat(messages)
        return answer
      except Exception as e:
        print(e)
