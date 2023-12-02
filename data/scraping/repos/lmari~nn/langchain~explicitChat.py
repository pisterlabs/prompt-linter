## langchain: un semplice esempio di chat con gpt
# Luca Mari, maggio 2023  
# [virtenv `langchain`: langchain, openai]

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

llm = ChatOpenAI(model_name='gpt-4', temperature=0, client='mychat')

messages:list = [SystemMessage(content='Sei un assistente')]

print('Buongiorno')

while True:
    user_prompt = input('\n>>> ')
    messages.append(HumanMessage(content=user_prompt))
    ai_response = llm(messages)
    print('\nAssistente: ', ai_response.content)
    messages.append(AIMessage(content=ai_response.content))
