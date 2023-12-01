import sys
import requests
import json
import re
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.chains.base import Chain
from langchain.input import get_colored_text

from typing import Any, Callable, Dict, List, Mapping, Optional, Union
from pydantic import root_validator

llm_decider = OpenAI(temperature=0.0)
llm_creative = OpenAI(temperature=0.7)
product_prompt = "What is a good name for a company that makes {product}?"


class LazyPrompt(PromptTemplate):
  def format(self, **kwargs: Any) -> str:
    prompt_template = kwargs[self.template]
    template_kwargs = dict(kwargs)
    template_kwargs.pop(self.template)
    return prompt_template.format(**template_kwargs)
  
  @root_validator()
  def template_is_valid(cls, values: Dict) -> Dict:
    return values


class CategorizationConditional(Chain):
  categorization_input: str
  subchains: Dict[str, Chain]
  default_chain: Chain
  output_variables: List[str] = ["text"]

  # TODO: validator requires the union of subchain inputs
  # TODO: validator requires all subchains have all output keys that are not also input keys

  @property
  def input_keys(self) -> List[str]:    
    return self.default_chain.input_keys + [key for subchain in self.subchains.values() for key in subchain.input_keys]

  @property
  def output_keys(self) -> List[str]:
    return self.output_variables

  def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
    categorization = inputs[self.categorization_input].strip()

    _colored_text = get_colored_text(categorization, "yellow")
    _text = "Categorization input:\n" + _colored_text
    self.callback_manager.on_text(_text, end="\n", verbose=self.verbose)

    subchain = self.subchains[categorization] if categorization in self.subchains else self.default_chain
 
    known_values = inputs.copy()
    outputs = subchain(known_values, return_only_outputs=True)
    known_values.update(outputs)
    return {k: known_values[k] for k in self.output_variables}    


class ESearchChain(Chain):
  output_variable: str = "text"
  input_variable: str = "text"

  @property
  def input_keys(self) -> List[str]:    
    return [self.input_variable]

  @property
  def output_keys(self) -> List[str]:
    return [self.output_variable]

  def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
    search = inputs[self.input_variable]
    search = 'x-ray'
    url = f"http://localhost:9200/coverages/_search?q={requests.utils.quote(search)}"
    response = requests.get(url)
    decoded_response = json.loads(response.content)

    return {self.output_variable: decoded_response['hits']['hits']}


format_hit = lambda i, h: f"{i+1}: ${h['rate']} for {h['name']} {h['description']} (CPT code: {h['billing_code']})."


search_format_chain = TransformChain(
  input_variables=['hits'],
  output_variables=['search_results'],
  transform = lambda inputs: {'search_results': '\n'.join([format_hit(i, h['_source']) for (i, h) in enumerate(inputs['hits'])])}
)

coverage_question_chain = SequentialChain(
  input_variables=["disambiguation_prompt", "response_prompt", "response_prompt", "identity", "context", "response_prefix", "user_input"],
  chains=[
    LLMChain(llm=llm_decider, prompt=LazyPrompt(input_variables=["disambiguation_prompt", "identity", "context", "user_input"], template="disambiguation_prompt"), output_key="search", verbose=True),
    ESearchChain(input_variable="search", output_variable="hits"),
    search_format_chain,
    LLMChain(llm=llm_creative, prompt=LazyPrompt(input_variables=["search", "response_prompt", "search_results", "context", "identity", "response_prefix"], template="response_prompt"), output_key="response", verbose=True),
    TransformChain(
      input_variables=["response_prefix", "response"], output_variables=["text"], 
      transform = lambda inputs: {'text': inputs['response_prefix'] + inputs['response']}
    )
  ]
)

chain = SequentialChain(
  input_variables=["categorization_prompt", "disambiguation_prompt", "response_prompt", "default_prompt", "response_prefix", "identity", "context", "user_input"],
  chains=[
    LLMChain(llm=llm_decider, prompt=LazyPrompt(input_variables=["categorization_prompt", "context", "user_input"], template="categorization_prompt"), output_key="categorization", verbose=True),
    CategorizationConditional(
      input_variables=["disambiguation_prompt", "response_prompt", "default_prompt", "response_prefix", "identity", "context", "user_input"],
      categorization_input="categorization",
      verbose=True,
      subchains={
        "Questions about whether insurance will cover a medical procedure or service": coverage_question_chain,
        "Questions about how much insurance will cover for a medical procedure or service": coverage_question_chain,
      },
      default_chain=LLMChain(llm=llm_creative, prompt=LazyPrompt(input_variables=["default_prompt", "identity", "context", "user_input"], template="default_prompt"), verbose=True),
    ),
  ]
)

identity_txt = '''You are a very friendly, positive and helpful representative of a health insurance company that is a good listener. 
The customer is enrolled in the company's insurance plan.'''

categorization_template = '''Context: ${context}\n The following is a list of categories
that insurance customer questions fall into:

Questions about whether insurance will cover a medical procedure or service, Questions about how much insurance will cover for a medical procedure or service, Other statements or questions.

{user_input}

Category:'''

disambiguation_template = '''{identity}\n{context}\nThe customer asks:\n\n{user_input}\n
What medical service, procedure or device is the customer asking about? Be concise.\n\nThe customer is asking about'''

response_template = '''Context: {context} Customer has insurance from a company that covers medical services at the following rates:\n
{search_results}
{identity} You will only answer about the services listed above.\n
Respond with normal capitalization. Use full words rather than abbrieviations. Do not provide CPT or medical codes in your responses. 
Do not respond with different coverage rates. 
Ask the customer to be more specific.
\n##\nCustomer: Tell me how much money my insurance covers for {search} and describe that service.\nAgent: {response_prefix}'''

default_template = '''{identity}\n{context}\n
You will only answer questions about health insurance.
\n##\nCustomer: ${user_input}\nAgent:'''

def prompt_from_template(template) -> PromptTemplate:
    keys = re.findall(r'{([^{}]*)}', template)
    return PromptTemplate(
      input_variables=keys,
      template = template
    )

system_inputs = {
  "identity": identity_txt,
  "context": "",
  "response_prefix": "Your insurance will",
  "categorization_prompt": prompt_from_template(categorization_template),
  "disambiguation_prompt": prompt_from_template(disambiguation_template),
  "response_prompt": prompt_from_template(response_template),
  "default_prompt": prompt_from_template(default_template),
}

user_inputs = {
  "user_input": sys.argv[1],
}

# Run the chain only specifying the input variable.
print(chain.run(dict(**system_inputs, **user_inputs)))