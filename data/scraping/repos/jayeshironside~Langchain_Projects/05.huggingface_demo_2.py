import os
from dotenv import load_dotenv
# Use the environment variables to retrieve API keys
load_dotenv()
HUGGINGFACEHUB_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

from langchain import HuggingFaceHub, PromptTemplate, LLMChain

template = """Question: {question}
Answer: Let's think step by step."""

# repo_id = "google/flan-t5-large"
repo_id = "google/flan-t5-xxl"
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt,llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.1, "max_length": 64}
    )
)

our_query = "What is COVID19 ?"
response = llm_chain.run(our_query)
print(response)
