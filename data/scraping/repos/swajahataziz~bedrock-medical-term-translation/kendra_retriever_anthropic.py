from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock

import os

import json
import sys

import boto3

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww


boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

def build_chain():
    region = os.environ["AWS_REGION"]
    kendra_index_id = os.environ["KENDRA_INDEX_ID"]
    
    # - create the Anthropic Model
    llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':512})
        
    retriever = AmazonKendraRetriever(index_id=kendra_index_id,region_name=region)

    prompt_template = """

      \n\nHuman: You are an AI assistant helping patients understand medical terminologies. 
      You are talkative and provide specific details from the context but limits it to 2000 tokens.
      If you do not know the answer to a question, you truthfully says you 
      do not know.

      \n\nAssistant: OK, got it, I'll be a talkative truthful AI assistant.

      \n\nHuman: Here are a few resources in <documents> tags:
      <documents>
      {context}
      </documents>
      Based on the above context, provide a detailed answer for, {question} Answer "don't know" 
      if not present in the resources provided. Start your answer with, "Based on the information provided from some trusted sources..." 

      \n\nAssistant:"""

    PROMPT = PromptTemplate(
          template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    return RetrievalQA.from_chain_type(
      llm, 
      chain_type="stuff", 
      retriever=retriever, 
      chain_type_kwargs=chain_type_kwargs,
      return_source_documents=True
  )

def run_chain(chain, prompt: str, history=[]):
    result = chain(prompt)
    # To make it compatible with chat samples
    return {
        "answer": result['result'],
        "source_documents": result['source_documents']
    }

if __name__ == "__main__":
    chain = build_chain()
    result = run_chain(chain, "What's SageMaker?")
    print(result['answer'])
    if 'source_documents' in result:
        print('Sources:')
        for d in result['source_documents']:
            print(d.metadata['source'])
