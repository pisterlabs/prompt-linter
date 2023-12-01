from langchain import PromptTemplate, OpenAI, LLMChain
from model.openai import llm

prompt_template = "What is a good name for a company that makes {product}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)
