
ELECTRONIC_CONDITION_CONST = \
{"inputs":["product", "n_top"],
 "outputs": {"questions": """a js array of elements. Each element should contains 2 properties:
 question: str // the question.
 options: str // a js array of answers for the question. The array's length must not be greater than 3."""},
"template": """You are inspecting a secondhand {product}. Given a list of key points which are delimited by triple backquotes.

```
1. Noticeable malfunctions.
2. Physical damages.
3. Valid warranty .
```

What questions would you ask to gain more information for the given list of key points. Please list out {n_top} questions.

{format_instructions}.
Questions:"""}


from langchain.llms import BaseLLM
from langchain import LLMChain
import sys
import os
sys.path.append(f"{os.path.dirname(__file__)}/../..")
from botcore.utils.prompt_utils import build_prompt

def build_ask_electronic_condition_chain(model: BaseLLM):
    """
    Chain designed to make questions about a product's condition
    Input: chain({"product": "rice cooker", "n_top": 5})
    """
    inputs = ELECTRONIC_CONDITION_CONST['inputs']
    outputs = ELECTRONIC_CONDITION_CONST['outputs']
    template = ELECTRONIC_CONDITION_CONST['template']
    prompt = build_prompt(inputs, outputs, template, include_parser=False)
    chain = LLMChain(llm=model, prompt=prompt, output_key='result')
    return chain
