from langchain.llms import OpenAI

# Accessing the OPENAI KEY
import environ
env = environ.Env()
environ.Env.read_env()
API_KEY = env('OPENAI_API_KEY')

# Creating a prompt template and running the LLM chain
from langchain import PromptTemplate, LLMChain
template = "What are the top {n} resources to learn {language} programming?"
prompt = PromptTemplate(template=template,input_variables=['n','language'])
llm = OpenAI(model_name="text-davinci-003", openai_api_key=API_KEY)
chain = LLMChain(llm=llm,prompt=prompt)
input = {'n':3,'language':'Python'}
print(chain.run(input))