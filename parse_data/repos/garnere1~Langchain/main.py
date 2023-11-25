import os
import credentials
os.environ["OPENAI_API_KEY"] = credentials.api_key
os.environ["SERPAPI_API_KEY"] = credentials.serpapi_key

from langchain import ConversationChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (PromptTemplate, 
                               ChatPromptTemplate, 
                               MessagesPlaceholder, 
                               SystemMessagePromptTemplate, 
                               HumanMessagePromptTemplate)
from langchain.chains import LLMChain
from langchain.agents import (load_tools, 
                              initialize_agent, 
                              AgentType)
from langchain.schema import (AIMessage, 
                              HumanMessage, 
                              SystemMessage)
from langchain.prompts.chat import (ChatPromptTemplate, 
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain.memory import ConversationBufferMemory


llm = OpenAI(temperature=0.9)
prompt=PromptTemplate(
    input_variables=["topic"],
    template="name one movie about {topic}"
)
chain = LLMChain(llm=llm, prompt=prompt)
val = input("Enter a topic: ")
print(chain.run(val))
