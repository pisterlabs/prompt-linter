# Testing integration between Langchain and Text Generation UI
# This example is taken from 
# https://python.langchain.com/docs/modules/model_io/models/llms/integrations/textgen

model_url = "http://localhost:5000"
import langchain
from langchain import PromptTemplate, LLMChain
from langchain.llms import TextGen

langchain.debug = True

template = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
### Human: {question}

### Assistant:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = TextGen(model_url=model_url)
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What is the capital of France?"

llm_chain.run(question)