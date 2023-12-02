ASK_CONDITION_CONST = \
{"inputs":["product", "n_top"],
 "outputs": {"chain": "always return 'ask_condition'.", "questions": """a js array of elements. Each element should contains 2 properties:
 question: str // the question.
 options: str // a js array of answers for the question. The array's length must not be greater than 5."""},
"template": """You are inspecting a secondhand {product}.
Please come up with exactly {n_top} common questions that will allow you to gather more information about the following criteria, which are delimited by triple backquotes.

```
* Any malfunctions, defections.
* Current physical condition.
* Check warranty if the product is an electronic device.
```

{format_instructions}.
Questions:"""}

from langchain.llms import BaseLLM
from langchain import LLMChain
import sys
import os
sys.path.append(f"{os.path.dirname(__file__)}/../..")
from botcore.utils.prompt_utils import build_prompt

def build_ask_condition_chain(model: BaseLLM):
    """
    Chain designed to make questions about a product's condition
    Input: chain({"product": "rice cooker", "n_top": 5})
    """
    inputs = ASK_CONDITION_CONST['inputs']
    outputs = ASK_CONDITION_CONST['outputs']
    template = ASK_CONDITION_CONST['template']
    prompt = build_prompt(inputs, outputs, template, include_parser=False)
    chain = LLMChain(llm=model, prompt=prompt, output_key='result')
    return chain
