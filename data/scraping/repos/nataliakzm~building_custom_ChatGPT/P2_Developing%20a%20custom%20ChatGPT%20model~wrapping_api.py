import os 
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

#Use your own API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#Define the instructions for the LLM
template = """Question: {question}
           Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm= OpenAI())

#Ask a question
question = "Which city hosted the Summer Olympics in the year the Titanic sank?"

llm_chain.run(question)