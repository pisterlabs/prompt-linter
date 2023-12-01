
import os
from dotenv import load_dotenv
import openai

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set the OpenAI API key
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

template = """Answer the following question in the tone of {tone} including emojis

question: {question}
 """



def initialise_llm(tone:str, question:str)->str:
  """initialise llm model at the start of every conversation"""
  prompt = PromptTemplate(template=template, input_variables=["tone","question"])
  llm = OpenAI(openai_api_key=openai.api_key)
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  answer = llm_chain.run(tone=tone, question=question)

  return answer


conversation_history = ""


while True:

    user_input = input("You: ")


    if user_input.lower() == "exit":
        break


    conversation_history = f"You: {user_input}\n"


    response = initialise_llm(tone="sarcastic 14 year old girl", question=conversation_history)



    print(f"Bot: {response}\n")


