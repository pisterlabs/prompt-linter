from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from dotenv import load_dotenv
load_dotenv()

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.1})
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who is your creator?"
print(question)
print('➡️ ', llm_chain.run(question))
