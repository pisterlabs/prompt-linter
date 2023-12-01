### System Imports
import os
import sys

### LC Prompts Imports
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

### LLM Chains Imports
from langchain import LLMChain
from langchain.chains import ConversationChain, ConversationalRetrievalChain

### LLM Models Imports
from langchain.chat_models import ChatOpenAI

### LLM Memory Imports
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = 


### LLMS
chat = ChatOpenAI(temperature=0)





### Templates
system_template = "Du bist der Kundendienst Bot von SuperSiuu. Die Frage lautet: {question}"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "Wie groß ist der mond?"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

first_q = PromptTemplate(template=system_message_prompt, 
                           input_variables=["question"]
                           )





chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                 human_message_prompt,
                                                 MessagesPlaceholder(variable_name="history")])



### Memory
memory = ConversationBufferMemory(return_messages=True)


conversation = ConversationChain(memory=memory, prompt=chat_prompt, llm=chat)
result = conversation.predict(input=first_q)
print(result)


# follow up question
follow_up = "Und wie weit ist er weg?"
result2 = conversation.predict(input=follow_up)
print(result2)
