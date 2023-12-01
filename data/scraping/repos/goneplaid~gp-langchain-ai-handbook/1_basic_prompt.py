from langchain.llms import OpenAI
import os

# initialize the models
openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key=os.environ['OPENAI_API_KEY']
)

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

print(openai(prompt))
