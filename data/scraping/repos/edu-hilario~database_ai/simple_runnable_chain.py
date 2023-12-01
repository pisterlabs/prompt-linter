from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser

import environ

env = environ.Env()
environ.Env.read_env()
API_KEY = env("OPENAI_API_KEY")


llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY)
prompt = PromptTemplate.from_template(
    "What are the top {n} resources to learn {language} programming?"
)
output_parser = StrOutputParser()
runnable = prompt | llm | output_parser
variable_dictionary = {"language": "python", "n": 5}
result = runnable.invoke(variable_dictionary)
print(result)
