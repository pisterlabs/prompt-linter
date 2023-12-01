from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os
import webbrowser
import urllib
        

os.environ["OPENAI_API_KEY"] = open("./api_key.txt", "r").read().strip()
def read_agent():
    text_file = open("./vis_agent.txt", "r")

    #read whole file to a string
    data = text_file.read()
    
    #close file
    text_file.close()
    
    return data

agent_prompt = read_agent()

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(agent_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0.7)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

text_file = open("./document.txt", "r")
data = text_file.read()
text_file.close()

while(True):
    k = memory.load_memory_variables({})

    print('document:\n')
    print(data)
    print('prompt:\n')
    user_prompt = input()
    conversation.predict(input=f"documents:\n${data}\n\nuser_prompt:\n${user_prompt}")

    last_ai_message = k['history'][-1].content
    print(last_ai_message)