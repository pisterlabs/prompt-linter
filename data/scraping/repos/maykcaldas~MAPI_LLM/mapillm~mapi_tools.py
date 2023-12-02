from mp_api.client import MPRester
from emmet.core.summary import HasProps
import openai
import langchain
from langchain import OpenAI
from langchain import agents
from langchain.agents import initialize_agent
from langchain.agents import Tool, tool
from langchain import LLMMathChain, SerpAPIWrapper
from gpt_index import GPTListIndex, GPTIndexMemory
from langchain import SerpAPIWrapper
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import (MaxMarginalRelevanceExampleSelector, 
                                                SemanticSimilarityExampleSelector)
import requests
from rdkit import Chem
import pandas as pd
import os

class MAPITools:
  def __init__(self):
    self.model = 'text-ada-001' #maybe change to gpt-4 when ready
    self.k=10
  
  def get_material_atoms(self, formula):
    '''Receives a material formula and returns the atoms symbols present in it separated by comma.'''
    import re
    pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
    matches = pattern.findall(formula)
    atoms = []
    for m in matches:
      atom, count = m
      count = int(count) if count else 1
      atoms.append((atom, count))
    return ",".join([a[0] for a in atoms])

  def check_prop_by_formula(self, formula):
    raise NotImplementedError('Should be implemented in children classes')

  def search_similars_by_atom(self, atoms):
    '''This function receives a string with the atoms separated by comma as input and returns a list of similar materials'''
    atoms = atoms.replace(" ", "")
    with MPRester(os.getenv("MAPI_API_KEY")) as mpr:
      docs = mpr.summary.search(elements=atoms.split(','), fields=["formula_pretty", self.prop])
    return docs

  def create_context_prompt(self, formula):
    raise NotImplementedError('Should be implemented in children classes')

  def LLM_predict(self, prompt):
    ''' This function receives a prompt generate with context by the create_context_prompt tool and request a completion to a language model. Then returns the completion'''
    llm = OpenAI(
          model_name=self.model,
          temperature=0.7,
          n=1,
          best_of=5,
          top_p=1.0,
          stop=["\n\n", "###", "#", "##"],
          # model_kwargs=kwargs,
      )
    return llm.generate([prompt]).generations[0][0].text

  def get_tools(self):
    return [
        Tool(
            name = "Get atoms in material",
            func = self.get_material_atoms,
            description = (
              "Receives a material formula and returns the atoms symbols present in it separated by comma."
              )
        ),
        Tool(
            name = f"Checks if material is {self.prop_name} by formula",
            func = self.check_prop_by_formula,
            description = (
                f"This functions searches in the material project's API for the formula and returns if it is {self.prop_name} or not."
              )
        ),
        # Tool(
        #     name = "Search similar materials by atom",
        #     func = self.search_similars_by_atom,
        #     description = (
        #       "This function receives a string with the atoms separated by comma as input and returns a list of similar materials."
        #       )
        # ),
        Tool(
            name = f"Create {self.prop_name} context to LLM search",
            func = self.create_context_prompt,
            description = (
              f"This function received a material formula as input and create a prompt to be inputed in the LLM_predict tool to predict if the material is {self.prop_name}." 
              if isinstance(self, MAPI_class_tools) else
              f"This function received a material formula as input and create a prompt to be inputed in the LLM_predict tool to predict the {self.prop_name} of a material." 
              )
        ),
        Tool(name = "LLM predictiom",
            func = self.LLM_predict,
            description = (
                "This function receives a prompt generate with context by the create_context_prompt tool and request a completion to a language model. Then returns the completion"
              )
        )
    ]

class MAPI_class_tools(MAPITools):
  def __init__(self, prop, prop_name, p_label, n_label):
    super().__init__()
    self.prop = prop
    self.prop_name = prop_name
    self.p_label = p_label
    self.n_label = n_label

  def check_prop_by_formula(self, formula):
    f''' This functions searches in the material project's API for the formula and returns if it is {self.prop_name} or not'''
    with MPRester(os.getenv("MAPI_API_KEY")) as mpr:
      docs = mpr.summary.search(formula=formula, fields=["formula_pretty", self.prop])
    if docs:
      if docs[0].formula_pretty == formula:
        return self.p_label if docs[0].dict()[self.prop] else self.n_label
    return f"Could not find any material while searching {formula}"

  def create_context_prompt(self, formula):
    '''This function received a material formula as input and create a prompt to be inputed in the LLM_predict tool to predict if the formula is a {self.prop_name} material '''
    elements = self.get_material_atoms(formula)
    similars = self.search_similars_by_atom(elements)
    similars = [
        {'formula': ex.formula_pretty,
        'prop': self.p_label if ex.dict()[self.prop] else self.n_label
        } for ex in similars
    ]
    examples = pd.DataFrame(similars).drop_duplicates().to_dict(orient="records")
    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                    examples,
                    OpenAIEmbeddings(),
                    FAISS,
                    k=self.k,
                  )
    
    prefix=(
      f'You are a bot who can predict if a material is {self.prop_name}.\n'
      f'Given this list of known materials and the information if they are {self.p_label} or {self.n_label}, \n'
      f'you need to answer the question if the last material is {self.prop_name}:'
      )
    prompt_template=PromptTemplate(
                  input_variables=["formula", "prop"],
                  template=f"Is {{formula}} a {self.prop_name} material?@@@\n{{prop}}###",
              )
    suffix = f"Is {{formula}} a {self.prop_name} material?@@@\n"
    prompt = FewShotPromptTemplate(
              # examples=examples,
              example_prompt=prompt_template,
              example_selector=example_selector,
              prefix=prefix,
              suffix=suffix,
              input_variables=["formula"])
    
    return prompt.format(formula=formula)

class MAPI_reg_tools(MAPITools):
  # TODO: deal with units
  def __init__(self, prop, prop_name):
    super().__init__()
    self.prop = prop
    self.prop_name = prop_name

  def check_prop_by_formula(self, formula):
    ''' This functions searches in the material project's API for the formula and returns the {self.prop_name}'''
    with MPRester(os.getenv("MAPI_API_KEY")) as mpr:
      docs = mpr.summary.search(formula=formula, fields=["formula_pretty", self.prop])
    if docs:
      if docs[0].formula_pretty == formula:
        return docs[0].dict()[self.prop]
      elif docs[0].dict()[self.prop] is None:
        return f"There is no record of {self.prop_name} for {formula}"
    return f"Could not find any material while searching {formula}"

  def create_context_prompt(self, formula):
    f'''This function received a material formula as input and create a prompt to be inputed in the LLM_predict tool to predict the {self.prop_name} of the material '''
    elements = self.get_material_atoms(formula)
    similars = self.search_similars_by_atom(elements)
    similars = [
        {'formula': ex.formula_pretty,
        'prop': f"{ex.dict()[self.prop]:2f}" if ex.dict()[self.prop] is not None else None
        } for ex in similars
    ]
    examples = pd.DataFrame(similars).drop_duplicates().dropna().to_dict(orient="records")

    example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                    examples,
                    OpenAIEmbeddings(),
                    FAISS,
                    k=self.k,
                  )
    
    prefix=(
      f'You are a bot who can predict the {self.prop_name} of a material .\n'
      f'Given this list of known materials and the measurement of their {self.prop_name}, \n'
      f'you need to answer the what is the {self.prop_name} of the material:'
       'The answer should be numeric and finish with ###'
      )
    prompt_template=PromptTemplate(
                  input_variables=["formula", "prop"],
                  template=f"What is the {self.prop_name} for {{formula}}?@@@\n{{prop}}###",
              )
    suffix = f"What is the {self.prop_name} for {{formula}}?@@@\n"
    prompt = FewShotPromptTemplate(
              # examples=examples,
              example_prompt=prompt_template,
              example_selector=example_selector,
              prefix=prefix,
              suffix=suffix,
              input_variables=["formula"])
    
    return prompt.format(formula=formula)
