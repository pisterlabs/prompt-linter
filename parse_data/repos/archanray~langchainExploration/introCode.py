import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# default API doesn't allow using GPU, so use smaller models
# like:  google/flan-t5-x1

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI

# initialize HF LLM
# the parameter temperature is used in the logits function
# specifically to set this:
# qi = exp(zi/T) / \sum_j exp(zj/T), 
# where T is the temperature and z_i is the ith entry of the
# output vector (check a bit more on this)
# so higher T implies softer softmax outputs
# softer models are used to avoid having repeatative oputputs
flan_t5 = HuggingFaceHub(
  repo_id = "google/flan-t5-large",
  model_kwargs = {"temperature":1e-10, "max_length":128}
  )

# build prompt template for simple question-answering
template = """Question: {question}

            Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
    prompt=prompt,
    llm = flan_t5
    )

question = "Where is Brazil?"
# both small and large model fails terribly in finding intricate details
print(llm_chain.run(question)) 

## for multiple queries first make a list of dictionaries
qs = [
      {"question": "what is an LLM?"},
      {"question": "Can children play in parks?"},
      {"question": "How many eyes does a pencil have?"}
]
print(llm_chain.generate(qs))

## for sending multiple queries as a long chain of questions
multi_template= """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(
        template = multi_template,
        input_variables = ["questions"]
  )

llm_chain = LLMChain(
  prompt = long_prompt,
  llm = flan_t5
  )

qs_str = (
  "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?" +
    "How many eyes does a blade of grass have?"
  )
print(llm_chain.run(qs_str))