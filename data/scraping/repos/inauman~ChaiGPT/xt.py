from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from flask import Flask, Response, request, stream_with_context
from apikey import OPENAI_API_KEY
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json


chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
question = "What is bitcoin?"

# def chatx():
#     def generate():
#         yield chat([HumanMessage(content=question)])
#     return generate()

def chatx():
    #def generate():
    for chunk in chat([HumanMessage(content=question)]):
        print(chunk[1])
        

    #return generate()
#  
# write a function to call chat function and print the values as they are generated
def main():
    chatx()
   
if __name__ == "__main__":
    main()