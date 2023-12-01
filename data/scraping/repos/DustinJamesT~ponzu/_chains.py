

# -- langchain imports
from langchain.chains import LLMChain
from langchain.chains import ConversationalRetrievalChain, create_tagging_chain
from langchain.chains.summarize import load_summarize_chain

from langchain.prompts import PromptTemplate

from langchain.output_parsers import DatetimeOutputParser

# -- local imports 
from ._api import loadLLM


# ==================================================
# -- LLM CHAIN
# ==================================================

def loadLLMchain(prompt, temp = 0.3, model = '', output_parser = None, output_key = 'text'):
  # -- assumes prompt has been initialized with langchain PromptTemplate
  llm = loadLLM(model = model, temp=temp)
  chain = LLMChain(llm=llm, prompt=prompt, output_key=output_key) if output_parser == None else LLMChain(llm=llm, prompt=prompt, output_parser=output_parser, output_key=output_key)

  return chain

def generationChain(prompt, input_variables, model = '', temp = 0.3, output_key = 'text', output_parser = None):
  # -- initialize prompt 
  prompt_ = PromptTemplate(template=prompt, input_variables=input_variables)

  # -- initialize chain
  chain = loadLLMchain(prompt_, model=model, temp=temp, output_key=output_key, output_parser=output_parser)

  return chain 

# ==================================================
# -- Summarization
# ==================================================

def loadSummaryChain(llm, prompt = None, chain_type="map_reduce" , combine_prompt=None, return_intermediate_steps=False):
  
  # -- load chain
  if prompt == None:
    chain = load_summarize_chain(llm, chain_type=chain_type, return_intermediate_steps=return_intermediate_steps)
  else:
    # -- load prompt 
    combine_prompt = prompt if combine_prompt == None else combine_prompt
    chain = load_summarize_chain(llm, chain_type=chain_type, return_intermediate_steps=return_intermediate_steps, map_prompt=prompt, combine_prompt=combine_prompt)

  return chain

# ==================================================
# -- Functions
# ==================================================

def simpleLlmFunction(schema, model = 'chat', temp = 0.3): 
  llm = loadLLM(model = model, temp=temp)
  chain = create_tagging_chain(schema, llm)
  return chain

def dateChain(text, model = 'chat', temp = 0.3):
  llm = loadLLM(model = model, temp=temp)

  output_parser = DatetimeOutputParser()

  template = """Answer the users question:

  {question}

  {format_instructions}"""

  prompt = PromptTemplate.from_template(
      template,
      partial_variables={"format_instructions": output_parser.get_format_instructions()},
  )

  chain = LLMChain(prompt=prompt, llm=llm)

  output = chain.run("What date is mentioned in the text? #### TEXT #### " + text + " #### END TEXT ####")

  date_output = output_parser.parse(output)

  return date_output




