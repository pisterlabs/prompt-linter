from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

llm = OpenAI()
chat_model = ChatOpenAI()

#####################

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(openai_api_key="...")

#####################

# prompts.py
from langchain.prompts import PromptTemplate

description_prompt = PromptTemplate.from_template(
    "Write me a description for a TikTok about {topic}")

#####################

#from langchain.schema import HumanMessage

#text = "What would be a good company name for a company that makes colorful socks?"
#messages = [HumanMessage(content=text)]

#llm.invoke(text)
# >> Feetful of Fun

#chat_model.invoke(messages)
# >> AIMessage(content="Socks O'Color")
