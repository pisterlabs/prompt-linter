from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])
llm = OpenAI(model_name="text-davinci-003", temperature=.95)
chain = LLMChain(llm=llm, prompt=prompt)

output = chain.run("I am creating a character for a science fiction RPG, can you give me three possible names and short backstories for my character?")
print(output)


