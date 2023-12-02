from langchain.prompts import FewShotChatMessagePromptTemplate
# Import Azure OpenAI
from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain, TransformChain
from langchain.prompts import FewShotChatMessagePromptTemplate
import os

def set_env():
    pass

set_env()

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chat_models import AzureChatOpenAI

def get_vehicle_details(inputs: dict) -> dict:
    text = inputs["intent"]
    return {"user_vehicle_details" : text}
    
def fetch_user_vehicle_details_chain():
    uvd_chain = TransformChain(input_variables=["intent"], output_variables=["user_vehicle_details"], transform=get_vehicle_details)
    return uvd_chain
