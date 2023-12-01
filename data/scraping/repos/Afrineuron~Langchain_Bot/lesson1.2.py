#Lesson 1.2: Using prompt templates



#Importing the required libraries
from langchain.llms.openai import OpenAI        #This is the import for the llm(flat-model)
from langchain.chat_models import ChatOpenAI    #This is the import for the chat model         
from langchain.prompts import PromptTemplate    #This is the import for the prompt template
from dotenv import load_dotenv                  #This is the import that helps load the dotenv(load_dotenv)  
import openai
import os      


#These lines are used to set the openai api key.
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]


#These lines are used to initialize an llm model(flat model) and a chat model.
llm = OpenAI()
chat_model = ChatOpenAI()


# prompt = PromptTemplate.from_template("Please write a quote about {day} that encourages {name} to accomplish all their todos.")
prompt = PromptTemplate.from_template("Please write a short quote about {day} that encourages {name} to accomplish all their todos. Be sure to address them personally.")
prompt = prompt.format(day="Monday", name="Louis")



#Run the flat model and print the response
llm_response = llm.predict(prompt)
print(llm_response)


#Insert some spaces between the flat model response and the chat model response.
print("\n\n\n")


#Run the chat model and print the response
chat_response = chat_model.predict(prompt)
print(chat_response)