from flask import Flask
from flask import request

import boto3
from langchain.retrievers import AmazonKendraRetriever
from typing import List
from typing import Dict
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#from langchain import SagemakerEndpoint, LLMChain
from langchain import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
import json

from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

import logging 

logging.basicConfig(filename='logs/api-access.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
app = Flask(__name__)




@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())



@app.route('/bedrock-anthropic-claude-v2', methods=['POST'])
def claude_v2():
    
    topk = request.args.get('topk')
    query = request.get_data().decode('utf-8')
    query = query.strip()
    #print(query)
    
    
    if "topk" in request.args:
        topk = int(request.args.get('topk'))
    else:
        topk = 250

    if "maxTokensToSample" in request.args:
        maxTokensToSample = int(request.args.get("maxTokensToSample"))
    else :
        maxTokensToSample= 300

   
    if "temperature" in request.args:
        temperature = float(request.args.get("temperature"))
    else :
        temperature= 0.5
     
    if "topP" in request.args:
        topP = float((request.args.get("topP")))
    else:
        topP = 0.5
    
    
    modelArgs = {'max_tokens_to_sample': int(maxTokensToSample), 'temperature':float(temperature), "top_k":int(topk),"top_p": float(topP),"stop_sequences":[]}
    #modelArgs = {'max_tokens_to_sample': int(maxTokensToSample), 'temperature':float(temp), "top_k":int(topK),"top_p": float(topP),"stop_sequences":[]}
    llm2 = Bedrock(model_id="anthropic.claude-v2",model_kwargs=modelArgs)
    
    app.logger.info("args: " + json.dumps(modelArgs))
    app.logger.info("query: " + query)
    #query = "Human:" + query + "\nAssistant:"
    
    #output = llm2.predict(query)
    
    prompt_template = """Human: {context}{question}
    Assistant:"""
  
    PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
    )
  
    chain = load_qa_chain(llm=llm2, prompt=PROMPT)
  
    docs = ""
  
    app.logger.info("query:" + query)
    app.logger.info(modelArgs)
  
    output = chain({"input_documents":docs, "question": query}, return_only_outputs=False)

    return(output["output_text"])



@app.route('/bedrock-anthropic-claude-instant-v1', methods=['POST'])
def claude_instant():
    
    topk = request.args.get('topk')
    query = request.get_data().decode('utf-8')
    query = query.strip()
    #print(query)
    
    
    if "topk" in request.args:
        topk = int(request.args.get('topk'))
    else:
        topk = 250

    if "maxTokensToSample" in request.args:
        maxTokensToSample = int(request.args.get("maxTokensToSample"))
    else :
        maxTokensToSample= 300

   
    if "temperature" in request.args:
        temperature = float(request.args.get("temperature"))
    else :
        temperature= 0.5
     
    if "topP" in request.args:
        topP = float((request.args.get("topP")))
    else:
        topP = 0.5
    
    
    modelArgs = {'max_tokens_to_sample': int(maxTokensToSample), 'temperature':float(temperature), "top_k":int(topk),"top_p": float(topP),"stop_sequences":[]}
    #modelArgs = {'max_tokens_to_sample': int(maxTokensToSample), 'temperature':float(temp), "top_k":int(topK),"top_p": float(topP),"stop_sequences":[]}
    llm2 = Bedrock(model_id="anthropic.claude-instant-v1",model_kwargs=modelArgs)
    
    app.logger.info("args: " + json.dumps(modelArgs))
    app.logger.info("query: " + query)
    #query = "Human:" + query + "\nAssistant:"
    
    #output = llm2.predict(query)
    
    prompt_template = """Human: {context}{question}
    Assistant:"""
  
    PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
    )
  
    chain = load_qa_chain(llm=llm2, prompt=PROMPT)
  
    docs = ""
  
    app.logger.info("query:" + query)
    app.logger.info(modelArgs)
  
    output = chain({"input_documents":docs, "question": query}, return_only_outputs=False)

    return(output["output_text"])



@app.route('/bedrock-anthropic', methods=['POST'])
def bedrock_anthropic():
    
    topk = request.args.get('topk')
    query = request.get_data().decode('utf-8')
    query = query.strip()
    #print(query)
    
    
    if "topk" in request.args:
        topk = int(request.args.get('topk'))
    else:
        topk = 250

    if "maxTokensToSample" in request.args:
        maxTokensToSample = int(request.args.get("maxTokensToSample"))
    else :
        maxTokensToSample= 300

   
    if "temperature" in request.args:
        temperature = float(request.args.get("temperature"))
    else :
        temperature= 0.5
     
    if "topP" in request.args:
        topP = float((request.args.get("topP")))
    else:
        topP = 0.5
    
    
    modelArgs = {'max_tokens_to_sample': int(maxTokensToSample), 'temperature':float(temperature), "top_k":int(topk),"top_p": float(topP),"stop_sequences":[]}
    #modelArgs = {'max_tokens_to_sample': int(maxTokensToSample), 'temperature':float(temp), "top_k":int(topK),"top_p": float(topP),"stop_sequences":[]}
    llm2 = Bedrock(model_id="anthropic.claude-v1",model_kwargs=modelArgs)
    
    app.logger.info("args: " + json.dumps(modelArgs))
    app.logger.info("query: " + query)
    #query = "Human:" + query + "\nAssistant:"
    
    #output = llm2.predict(query)
    
    prompt_template = """Human: {context}{question}
    Assistant:"""
  
    PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
    )
  
    chain = load_qa_chain(llm=llm2, prompt=PROMPT)
  
    docs = ""
  
    app.logger.info("query:" + query)
    app.logger.info(modelArgs)
  
    output = chain({"input_documents":docs, "question": query}, return_only_outputs=False)

    return(output["output_text"])
    


app.run(host='0.0.0.0', port=1210)
