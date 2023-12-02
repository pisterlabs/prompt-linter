from typing import Annotated, Callable, Dict, List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.prompts import PromptTemplate
from lib.model.lang_chain_context import LangChainContext
from lib.chains.case_chain import CaseChain
from lib.chains.api_chain import APIChain
from lib.chains.reformat_chain import ReformatChain
from lib.chains.recording_chain import RecordingChain
from lib.chains.vector_search_chain import VectorSearchChain

ChainSpec = Annotated[Union[
  "APIChainSpec",
  "SequentialChainSpec",
  "LLMChainSpec",
  "CaseChainSpec",
  "ReformatChainSpec",
  "TransformChainSpec",
  "VectorSearchChainSpec"
  ], Field(discriminator='chain_type')]


class BaseChainSpec(BaseModel):
  chain_id: int

  @property
  def children(self) -> List[ChainSpec]:
    return []

  def to_lang_chain(self, ctx: LangChainContext) -> Chain:
    raise NotImplementedError

  def traverse(self, fn: Callable[[ChainSpec], None]) -> None:
    fn(self)

    for child in self.children:
      child.traverse(fn)

  def find_by_chain_id(self, chain_id: int) -> Optional[ChainSpec]:
    if self.chain_id == chain_id:
      return self

    for child in self.children:
      result = child.find_by_chain_id(chain_id)

      if result is not None:
        return result

    return None

  def copy_replace(self, replace: Callable[[ChainSpec], ChainSpec]):
    return replace(self).copy(deep=True)

  def wrapChain(self, chain: Chain, ctx: LangChainContext) -> Chain:
    if ctx.recording:
      ret_chain = RecordingChain(chain_spec_id=self.chain_id, chain=chain)
      ctx.prompts.append(ret_chain)
      return ret_chain
    return chain


class LLMChainSpec(BaseChainSpec):
  input_keys: List[str]
  output_key: str
  chain_type: Literal["llm_chain_spec"] = "llm_chain_spec"
  prompt: str
  llm_key: str

  def to_lang_chain(self, ctx: LangChainContext) -> Chain:
    llm = ctx.llms.get(self.llm_key)

    if llm is None:
      raise ValueError(f"LLM with key {self.llm_key} not found in context")

    promptTemplate = PromptTemplate(template=self.prompt, input_variables=self.input_keys)

    ret_chain = LLMChain(llm=llm, prompt=promptTemplate, output_key=self.output_key)

    return self.wrapChain(ret_chain, ctx)


class SequentialChainSpec(BaseChainSpec):
  input_keys: List[str]
  output_keys: List[str]
  chain_type: Literal["sequential_chain_spec"] = "sequential_chain_spec"
  chains: List[ChainSpec]

  @property
  def children(self) -> List[ChainSpec]:
    return list(self.chains)

  def to_lang_chain(self, ctx: LangChainContext) -> Chain:
    chains = [chain.to_lang_chain(ctx) for chain in self.chains]

    ret_chain = SequentialChain(chains=chains, input_variables=self.input_keys, output_variables=self.output_keys)

    return self.wrapChain(ret_chain, ctx)

  def copy_replace(self, replace: Callable[[ChainSpec], ChainSpec]):
    sequential = replace(self).copy(deep=True, exclude={"chains"})
    sequential.chains = [chain.copy_replace(replace) for chain in self.chains]
    return sequential


class CaseChainSpec(BaseChainSpec):
  chain_type: Literal["case_chain_spec"] = "case_chain_spec"
  cases: Dict[str, ChainSpec]
  categorization_key: str
  default_case: ChainSpec

  @property
  def children(self) -> List[ChainSpec]:
    return list(self.cases.values()) + [self.default_case]

  def to_lang_chain(self, ctx: LangChainContext) -> CaseChain:
    subchains = {key: chain.to_lang_chain(ctx) for key, chain in self.cases.items()}
    case_chain = CaseChain(
      subchains=subchains,
      categorization_input=self.categorization_key,
      default_chain=self.default_case.to_lang_chain(ctx),
    )

    return self.wrapChain(case_chain, ctx)

  def copy_replace(self, replace: Callable[[ChainSpec], ChainSpec]):
    case_chain = replace(self).copy(deep=True, exclude={"cases"})
    case_chain.cases = {key: chain.copy_replace(replace) for key, chain in self.cases.items()}
    return case_chain


class ReformatChainSpec(BaseChainSpec):
  chain_type: Literal["reformat_chain_spec"] = "reformat_chain_spec"
  formatters: Dict[str, str]
  input_keys: List[str]

  def to_lang_chain(self, ctx: LangChainContext) -> Chain:
    chain = ReformatChain(formatters=self.formatters, input_variables=self.input_keys)
    return self.wrapChain(chain, ctx)


class TransformChainSpec(BaseChainSpec):
  chain_type: Literal["transform_chain_spec"] = "transform_chain_spec"
  transform_func: str
  input_keys: List[str]
  output_keys: List[str]

  def create_function(self, body):
    scope = {}
    indented = body.replace("\n", "\n  ")
    code = f"def f(inputs: dict):\n  {indented}"
    exec(code, scope)
    return scope["f"]

  def to_lang_chain(self, ctx: LangChainContext) -> Chain:
    chain = TransformChain(input_variables=self.input_keys, output_variables=self.output_keys, transform=self.create_function(self.transform_func))
    return self.wrapChain(chain, ctx)


class APIChainSpec(BaseChainSpec):
  chain_type: Literal["api_chain_spec"] = "api_chain_spec"
  url: str
  method: str
  headers: Optional[Dict[str, str]]
  body: Optional[str]
  input_keys: List[str]
  output_key: str

  def to_lang_chain(self, ctx: LangChainContext) -> Chain:
    chain = APIChain(
      url=self.url,
      method=self.method,
      headers=self.headers,
      body=self.body,
      input_variables=self.input_keys,
      output_variable=self.output_key,
    )
    return self.wrapChain(chain, ctx)


class VectorSearchChainSpec(BaseChainSpec):
  chain_type: Literal["vector_search_chain_spec"] = "vector_search_chain_spec"
  query: str
  embedding_engine: str
  embedding_params: Dict[str, Any]
  database: str
  database_params: Dict[str, Any]
  num_results: int
  min_score: float = 0.0
  input_variables: List[str] = []
  output_key: str = 'results'

  def to_lang_chain(self, ctx: LangChainContext) -> Chain:
    chain = VectorSearchChain(
      query=self.query,
      embedding_engine=self.embedding_engine,
      embedding_params=self.embedding_params,
      database=self.database,
      database_params=self.database_params,
      input_variables=self.input_variables,
      num_results=self.num_results,
      output_key=self.output_key,
      min_score=self.min_score,
    )

    self.wrapChain(chain, ctx)


SequentialChainSpec.update_forward_refs()
CaseChainSpec.update_forward_refs()
