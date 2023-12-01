from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatAnthropic as Anthropic
from langchain.llms.bedrock import Bedrock

import sys
import os

import json

import boto3

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww


boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

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

def build_chain():
    region = os.environ["AWS_REGION"]
    kendra_index_id = os.environ["KENDRA_INDEX_ID"]
    
    # - create the Anthropic Model
    llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':512})
    print("Using Kendra Index:"+ kendra_index_id +" in region:"+region)
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

    condense_qa_template = """
    \n\nHuman:Given the following conversation and a follow up question, rephrase the follow up question 
    to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    \n\nAssistant:"""
    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        condense_question_prompt=standalone_question_prompt, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt":PROMPT})
    return qa

def run_chain(chain, prompt: str, history=[]):
    return chain({"question": prompt, "chat_history": history})

if __name__ == "__main__":
    chat_history = []
    qa = build_chain()
    print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
    for query in sys.stdin:
        if (query.strip().lower().startswith("new search:")):
            query = query.strip().lower().replace("new search:","")
            chat_history = []
        elif (len(chat_history) == MAX_HISTORY_LENGTH):
            chat_history.pop(0)
    result = run_chain(qa, query, chat_history)
    chat_history.append((query, result["answer"]))
    print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)
    if 'source_documents' in result:
        print(bcolors.OKGREEN + 'Sources:')
        for d in result['source_documents']:
            print(d.metadata['source'])
    print(bcolors.ENDC)
    print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)
    print(">", end=" ", flush=True)
print(bcolors.OKBLUE + "Bye" + bcolors.ENDC)
