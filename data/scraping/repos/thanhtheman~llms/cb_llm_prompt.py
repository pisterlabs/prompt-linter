from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
prompt = ChatPromptTemplate.from_template("you are a {role} tell me a joke about {foo}")
prompt2 = ChatPromptTemplate.from_template("explain this joke to me: {result}")

# a simple one prompt version
# chain = prompt | model.bind(stop="ladder")
# Why did the stupid person bring a
#response = chain.content

# 2 prompts + adding a parser to format the output
#invoke = doing all the sequential steps, here we ask for a joke, get the result, using that result to ask for an explanation
result = prompt | model | StrOutputParser()
chain = result | model | StrOutputParser()
response = chain.invoke({"role": "sarcastic comedian", "foo": "a stupid person", "result": f"{result}"})
print(response)

"""
This joke plays on the double meaning of "on the house." In a bar, "on the house" means that the drinks are free 
or provided by the establishment.
However, the stupid person misinterprets it literally, thinking that the drinks are physically on the roof of the bar
They bring a ladder to reach the drinks but fail to understand the correct meaning.The joke highlights their lack of intelligence or wit.
"""