import os
from langchain import PromptTemplate, LLMChain, HuggingFaceHub

#os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HF_API_KEY'

template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=['question'])

# user question
question = "Could you give me the last president of the United States?"

# initialize Hub LLM
hub_llm = HuggingFaceHub(
                         repo_id='bigscience/bloom',
                         model_kwargs={'temperature':1e-10}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
                     prompt=prompt,
                     llm=hub_llm
)

# ask the user question
print(llm_chain.run(question))

