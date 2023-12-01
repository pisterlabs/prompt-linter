#Lesson 1.3: Using LECL(langchain expression language)



#Importing the required libraries
from langchain.llms.openai import OpenAI        #This is the import for the llm(flat-model)
from langchain.chat_models import ChatOpenAI    #This is the import for the chat model      
from dotenv import load_dotenv                  #This is the import that helps load the dotenv(load_dotenv)     
from langchain.prompts import PromptTemplate    #This is the import for the prompt template
import openai                                   
import os             

from langchain.schema.output_parser import StrOutputParser     #This is the import for the output parser. An output parser converts the output of the llm to a desired format such as a string or JSON.


#These lines are used to set the openai api key.
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]



#These lines are used to initialize an llm model(flat model) and a chat model.
llm = OpenAI()
chat_model = ChatOpenAI()



#Instead of using a plain text prompt, you can use a PromptTemplate prompt as below to allow for introduction of variables.
prompt = PromptTemplate.from_template("Please write a quote about {day} that encourages {name} to accomplish all their todos.")
prompt = PromptTemplate.from_template("Please write a short quote about {day} that encourages {name} to accomplish all their todos. Be sure to address them personally.")



#Construct a chain using the prompt, flat model(llm) and the output parser(this converts the output of the llm to a string)
flat_chain = prompt | llm | StrOutputParser()

#Run the flat model chain
chain_response = flat_chain.invoke({"day": "Monday", "name": "Louis"})
print(chain_response)



#Insert some spaces between the flat model response and the chat model response.
print("\n\n\n")



#Construct a chain using the prompt, chat model(llm) and the output parser(this converts the output of the llm to a string)
chat_chain = prompt | chat_model | StrOutputParser()

#Run the flat model chain
chain_response = chat_chain.invoke({"day": "Monday", "name": "Louis"})
print(chain_response)