import streamlit as st
from langchain.llms import OpenAI
import logging
import sys
import boto3
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit_authenticator as stauth


st.title('Fruitstand Support App - Using Claude/Bedrock')

from typing import List
from typing import Dict
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import AmazonKendraRetriever
from langchain import LLMChain
from langchain.llms.bedrock import Bedrock

import json

from langchain.docstore.document import Document

kendraIndexId = "063e46f7-1953-4503-a46c-72aa1ddf826f"
region = "us-east-1"

kendra_retriever = AmazonKendraRetriever(
    index_id= kendraIndexId
)



def get_remote_ip() -> str:
    """Get remote ip."""

    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None

    return session_info.request.remote_ip

class ContextFilter(logging.Filter):
    def filter(self, record):
        record.user_ip = get_remote_ip()
        return super().filter(record)


def init_logging():
    # Make sure to instanciate the logger only once
    # otherwise, it will create a StreamHandler at every run
    # and duplicate the messages

    # create a custom logger
    logger = logging.getLogger("claude-bedrock")
    logger.setLevel(logging.INFO)

    if logger.handlers:  # logger is already setup, don't setup again
        return
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # in the formatter, use the variable "user_ip"
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s [user_ip=%(user_ip)s] - %(message)s")
    handler = logging.FileHandler('claude-access.log')
    handler.setLevel(logging.INFO)
    handler.addFilter(ContextFilter())
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    


def generate_response(input_text,maxTokensToSample,temp, topK, topP, doRag, modelSelection):
  
  if (modelSelection=="Claude v1"):
    selectedModel = "anthropic.claude-v1"
    modelArgs = {'max_tokens_to_sample': int(maxTokensToSample), 'temperature':float(temp), "top_k":int(topK),"top_p": float(topP),"stop_sequences":[]}
  elif (modelSelection=="Claude v2"):
    selectedModel = "anthropic.claude-v2"
    modelArgs = {'max_tokens_to_sample': int(maxTokensToSample), 'temperature':float(temp), "top_k":int(topK),"top_p": float(topP),"stop_sequences":[]}
  elif (modelSelection == "Jurassic Jumbo Instruct"):
    selectedModel = "ai21.j2-jumbo-instruct"
    modelArgs = {'maxTokens': int(maxTokensToSample), 'temperature':float(temp), "topP": float(topP)}
  elif (modelSelection == "Jurassic Grande Instruct"):
    selectedModel = "ai21.j2-grande-instruct"
    modelArgs = {'maxTokens': int(maxTokensToSample), 'temperature':float(temp), "topP": float(topP)}
  
      
  llm2 = Bedrock(
    model_id= selectedModel,
    model_kwargs=modelArgs
  )
    
  
  llm_query= input_text

  prompt_template = """Human:
  {context}
  {question}
  
  Assistant:
  """
  
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )
  
  chain = load_qa_chain(llm=llm2, prompt=PROMPT)
  
  docs = ""
  
  if doRag:
    docs = kendra_retriever.get_relevant_documents(llm_query)
  
  logger.info("query:" + llm_query)
  logger.info("params "+ maxTokensToSample + " " + temp + " " + topK + " " + topP + " " + str(doRag) + " " +  selectedModel)
  
  output = chain({"input_documents":docs, "question": llm_query}, return_only_outputs=False)
  st.info(output['output_text'])
  st.subheader("RAG data obtained from Kendra")
  
  for doc in output['input_documents']:
    st.info(doc)
  

def main():
  
  
  with open('authConfig.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
  
  
  authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
  )

  name, authentication_status, username = authenticator.login('Login', 'main')
    
  if authentication_status:
    authenticator.logout('Logout', 'main', key='unique_key')
    st.write(f'Welcome *{name}*')
    st.title('Some content')
  

    with st.form('my_form'):
      doRag = st.checkbox("RAG - Kendra" , value=False)
      modelSelection = st.selectbox('Which LLM model would you like for inference?', ('Claude v1', 'Jurassic Grande Instruct', 'Jurassic Jumbo Instruct', "Claude v2"))
    
      
      maxTokensToSample = st.text_input("max tokens to sample", 300)
      temp = st.text_input("temperature", 0.5)
      topK = st.text_input("top_k - Does not apply for Jurassic models", 250)
      topP = st.text_input("top_p", 0.5)
      text = st.text_area('Enter your query:', 'How do I charge my iPhone?')
      submitted = st.form_submit_button('Submit')
      if submitted:
        generate_response(text, maxTokensToSample, temp, topK, topP, doRag, modelSelection)
        
    with st.sidebar:
      add_markdown= st.subheader('About the demo')
      add_markdown= st.markdown('This is a sample application that uses **Bedrock** with RAG using Kendra. Data for RAG is from the Apple support pages')
      add_markdown= st.markdown('You can ask questions like **:blue["my iphone screen is broken, how can I fix it"]** or **:blue["how do I change the wallpaper on my iphone"]**')
      add_markdown= st.markdown('**WARNING** This website is for demo purposes only and only publicly available information should be shared in the input prompts')

  elif authentication_status is False:
    st.error('Username/password is incorrect')
  elif authentication_status is None:
    st.warning('Please enter your username and password')
  

if __name__ == "__main__":
    init_logging()

    logger = logging.getLogger("claude-bedrock")
    main()