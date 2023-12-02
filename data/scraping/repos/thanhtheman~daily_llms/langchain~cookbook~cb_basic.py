from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
prompt = ChatPromptTemplate.from_template("please tell me a joke about a {role}")
chain = prompt | model | StrOutputParser()
response = chain.invoke({"role":"stupid plumber"})

print(response)
