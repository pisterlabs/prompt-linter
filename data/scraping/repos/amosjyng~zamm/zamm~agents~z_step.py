from abc import abstractmethod
from typing import Any, Dict, Optional

from langchain.prompts.base import StringPromptTemplate

from zamm.prompts.dummy import DummyPromptTemplate

from .step import StepOutput


class ZStepOutput(StepOutput):
    logger_template: StringPromptTemplate

    def log(self, **kwargs) -> str:
        return self._log(
            condensed=kwargs["condensed"],
            previous=kwargs["previous"],
            next=kwargs["next"],
        )

    def _log(
        self,
        condensed: bool,
        previous: Optional[StepOutput],
        next: Optional[StepOutput],
    ) -> str:
        return self.logger_template.format(**self.template_args)

    @classmethod
    @abstractmethod
    def from_chain_output(cls, output: Dict[str, Any]):
        """Construct the step from chain output"""

    @property
    @abstractmethod
    def template_args(self) -> Dict[str, str]:
        """Construct the dict used to render this output"""


class DummyStepOutput(ZStepOutput):
    logger_template: StringPromptTemplate = DummyPromptTemplate()

    @classmethod
    def from_chain_output(cls, output: Dict[str, Any]):
        """Construct the step from chain output"""
        raise NotImplementedError()

    @property
    def template_args(self) -> Dict[str, str]:
        """Construct the dict used to render this output"""
        raise NotImplementedError()
