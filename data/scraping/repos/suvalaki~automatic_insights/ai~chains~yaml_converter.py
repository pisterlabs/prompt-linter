from typing import List, Optional, Dict, Any, TypeVar, Type, Generic

from pydantic_yaml import YamlModel
from pydantic_computed import Computed, computed

from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)

from ai.converter import create_pydantic_yaml_validated_chain


InputPrompt = TypeVar("InputPrompt")
ValidatorT = TypeVar("ValidatorT")


class YamlConvertedChain(Generic[InputPrompt, ValidatorT], Chain):
    output_tp: Type[ValidatorT]

    llm: BaseLanguageModel
    prompt: InputPrompt

    chain: Computed[Chain]

    @computed("chain")
    def calculate_chain(
        llm: BaseLanguageModel,
        output_tp: Type[ValidatorT],
        prompt: InputPrompt,
        **kwargs
    ) -> Chain:
        return create_pydantic_yaml_validated_chain(llm, output_tp, prompt)

    @property
    def input_keys(self) -> List[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        return self.chain.output_keys

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return self.chain(inputs, run_manager=run_manager)
