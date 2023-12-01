## langchain: un semplice esempio di chat con gpt, con streaming della risposta
# Luca Mari, maggio 2023  
# [virtenv `langchain`: langchain, openai]

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(model_name='gpt-4', temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], client='mychat')
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory(), verbose=False)

print('Buongiorno')

while True:
    user_prompt = input('\n>>> ')
    ai_response = conversation.predict(input=user_prompt)
