from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])


llm = OpenAI(openai_api_key="YOUR_OPENAI_API_KEY")
llm_chain = LLMChain(prompt=prompt, llm=llm)


question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
result = llm_chain.run(question)

print(result)