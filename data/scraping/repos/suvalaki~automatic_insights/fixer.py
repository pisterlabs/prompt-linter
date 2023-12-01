from typing import Generic, List, TypeVar, Dict, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel
from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    Callbacks,
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from ai.validation.base import (
    ValidationSignal, Validation, Validator, ParalellValidator, SequentialValidator
)

# The type for the thing we are validating
InputT = TypeVar("InputT")



VAL = Validator[InputT] | ParalellValidator[InputT] | SequentialValidator[InputT]

class ValidationTreatmentStatus(Enum):
    FAILED = False 
    SUCCESS = True


class ValidationTreatment(BaseModel, Generic[InputT]):
    status: ValidationTreatmentStatus
    modified: InputT
    error: Optional[Exception] = None

    class Config:
        arbitrary_types_allowed=True


class ValidationTreater(Generic[InputT], Chain):

    validator: Validator[InputT]

    input_object_key:str = "input"
    input_validation_key:str = "validation"
    output_key:str = "treatment"

    @abstractmethod
    def _treat(self, _input_object: InputT, _validation: Validation[InputT], **kwargs) -> InputT:
        ...

    def treat(self, _input_object: InputT, _validation: Validation[InputT], **kwargs) -> ValidationTreatment[InputT]:
        try:
            modified = self._treat(_input_object, _validation, **kwargs)
            return ValidationTreatment(
                status=ValidationTreatmentStatus.SUCCESS, 
                modified=modified, 
            )
        except Exception as e:
            return ValidationTreatment(
                status=ValidationTreatmentStatus.FAILED, 
                modified=modified, 
                error=e,
            )

    @property
    def input_keys(self) -> List[str]:
        return [self.input_object_key, self.input_validation_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, ValidationTreatment[InputT]]:
        fixed = self.treat(inputs[self.input_object_key], inputs[self.input_validation_key], **inputs)
        return {self.output_key: fixed}


    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, ValidationTreatment[InputT]]:
        fixed = self.treat(inputs[self.input_object_key], inputs[self.input_validation_key], **inputs)
        return {self.output_key: fixed}