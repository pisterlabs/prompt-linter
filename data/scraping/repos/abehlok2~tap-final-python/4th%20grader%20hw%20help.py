import os
from langchain.schema import SystemMesage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.memory import ConversationBufferMemory

gpt4 = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.15,
    max_tokens=500,

)

palm_llm = ChatGooglePalm(
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.15,
)

class HwAssistant:
    gpt4 = gpt4
    palm_llm = palm_llm
    
