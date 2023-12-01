from aws_langchain.kendra_index_retriever import KendraIndexRetriever
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import os
import boto3
from utils import get_openai_api_key

# AWS Region
region_name = os.getenv('AWS_REGION', 'us-east-1') 
# Set SSM Parameter Store name for the OpenAI API key and the OpenAI Model Engine
API_KEY_PARAMETER_PATH = '/openai/api_key'
# Create an SSM client using Boto3
ssm = boto3.client('ssm', region_name=region_name)
# Get the OpenAI API key from the SSM Parameter Store
openai_api_key = get_openai_api_key(ssm_client=ssm, parameter_path=API_KEY_PARAMETER_PATH)

def build_chain():
    kendra_index_id = "1821eec9-a317-472a-a053-cd3401b79fe1"

    llm = OpenAI(batch_size=5, temperature=0, max_tokens=300, openai_api_key=openai_api_key)

    retriever = KendraIndexRetriever(kendraindex=kendra_index_id,
        awsregion=region_name,
        return_source_documents=True)

    prompt_template = """
    The following is a friendly conversation between a human and an AI. 
    The AI is talkative and provides lots of specific details from its context.
    If the AI does not know the answer to a question, it truthfully says it 
    does not know.
    {context}
    Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" if not present in the document. Solution:
    """
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
