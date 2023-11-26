import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# prompt consists of multiple components
# -- instructions:
# -- -- tell model what to do, typically how to use inputs
# -- external information or context
# -- -- additional info that can be entered manually into prompt
# -- -- retrieve via vector database:
# -- -- or some other means
# -- user input or query:
# -- -- input by user
# -- output indicator
# -- -- beginning of generated text, Can use keywords as 
# -- -- "import", or "Chatbot:", or etc.

prompt = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: Which libraries and model providers offer LLMs?

Answer: """

from langchain.llms import HuggingFaceHub

# initialize the models
flan_t5 = HuggingFaceHub(
  repo_id = "google/flan-t5-large",
  model_kwargs = {"temperature":1, "max_length":128}
  )

# output of the language model
print(flan_t5(prompt))

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: {query}

Answer: """

from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# initialize the models
flan_t5 = HuggingFaceHub(
  repo_id = "google/flan-t5-large",
  model_kwargs = {"temperature":1, "max_length":128}
  )

prompt_template = PromptTemplate(
    input_variables = ["query"],
    template = template
    )

# output of the language model
print(flan_t5(
    prompt_template.format(
        query="Which libraries and model provides offer LLMs?"
        ))
    )
