from typing import Protocol, Generic, TypeVar, Optional, List, Any, Tuple


T = TypeVar("T")


class ValidatePart(Protocol, Generic[T]):
    def validation(self, t: T) -> None:
        ...

    def explain_error(self, step: T, e: Exception) -> str:
        ...

    def validate(self, t: T) -> bool:
        return self.get_error(t) is None

    def get_error(self, t: T) -> Optional[Exception]:
        try:
            self.validation(t)
        except Exception as e:
            return e
        return None

    def explain(self, t: T) -> str:
        e = self.get_error(t)
        if e is not None:
            return self.explain_error(t)
        return ""

    def __call__(self, t: T) -> bool:
        return self.validation(t)


class ValidateAll(Protocol, Generic[T]):
    def validate(self, t: T) -> Tuple[bool, Any]:
        ...

    def explain(self, t: T, error: Any) -> str:
        ...

    pass


from typing import Dict

from langchain import PromptTemplate
from langchain.chains.base import Chain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)
from pydantic_computed import Computed, computed

InputPrompt = TypeVar("InputPrompt")
ValidatorT = TypeVar("ValidatorT")


VALIDATION_PROMPT = """
You will be presented with an input prompt, parsed outputs from the model, and a validation analysis. 
You are to recreate outputs for the input prompt but with the validation errors resolved. 

The input prompt follows: 
{prompt}

The parsed output follows:
{output}

The validation analysis follows:
{validation}

Please fix the response:
"""

VALIDATION_PROMPT_TEMPLATE = PromptTemplate(
    template=VALIDATION_PROMPT,
    input_variables=["prompt", "output", "validation"],
)


class ValidationChain(Generic[InputPrompt, ValidatorT], Chain):
    llm: BaseLanguageModel
    prompt: InputPrompt
    chain: Chain
    validator: ValidatorT
    max_retry: int = 5

    @property
    def input_keys(self) -> List[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        return self.chain.output_keys

    def _call_on_validation_error(
        self,
        error: Exception,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        pass

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        reply = self.chain(inputs, run_manager=run_manager)

        if not self.validator.validate(reply):
            error = self.validator.explain(reply)
