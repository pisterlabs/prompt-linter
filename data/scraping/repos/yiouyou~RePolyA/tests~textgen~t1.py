import os
_RePolyA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(_RePolyA)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from repolya.local.textgen import TextGen
from langchain.globals import set_debug

set_debug(True)


model_url = "http://127.0.0.1:5552"

template = """Question: {question}

Answer: Let us think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

llm = TextGen(
    model_url=model_url,
    temperature=0.01,
    top_p=0.9,
    seed=10,
    max_tokens=200,
    streaming=False,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)
llm_chain.run(question)

# curl http://127.0.0.1:5552/v1/completions -H "Content-Type: application/json" -d '{"prompt": "Question: What NFL team won the Super Bowl in the year Justin Bieber was born?\n\nAnswer: Let us think step by step.", "max_tokens": 200, "temperature": 0.01 ,"top_p": 0.9, "seed": 10, "stream": false}'

