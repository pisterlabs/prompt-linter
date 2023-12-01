from typing import Dict, List, Optional

from langchain import OpenAI, PromptTemplate
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLLM
from pydantic import BaseModel, root_validator

from core.text_generation import _prompts


class ChainWithLLM(Chain, BaseModel):
    _llm: Optional[BaseLLM] = None

    @root_validator
    def setup_llm(cls, values):  # noqa pylint: disable=no-self-argument
        llm = values.get("_llm")
        if llm is not None:
            return values
        model_name = values["feature_flags"].model
        temperature = values["feature_flags"].temperature
        if "gpt" in model_name:
            model = ChatOpenAI(
                client=None, model_name=model_name, temperature=temperature
            )
        else:
            model = OpenAI(
                client=None,
                temperature=temperature,
                model_name=model_name,
                frequency_penalty=0,
                presence_penalty=0,
            )
        values["_llm"] = model
        return values


class TextGenerationChainFeatureFlags(BaseModel):
    model: str = "gpt-3.5-turbo"
    temperature: float = 0
    prompt_template: str = _prompts.email_prompt


class SimpleTextGenerationChain(ChainWithLLM):
    output_key: str = "output"

    feature_flags: TextGenerationChainFeatureFlags = TextGenerationChainFeatureFlags()

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        prompt = PromptTemplate(
            input_variables=self.input_keys,
            template=self.feature_flags.prompt_template,
        )

        llm_chain = LLMChain(llm=self._llm, prompt=prompt)
        output = llm_chain.predict(**inputs)
        output = {self.output_key: output}
        return output

    @property
    def input_keys(self) -> List[str]:
        return [
            "from_name",
            "from_last_name",
            "to_name",
            "to_last_name",
            "formal_level",
            "urgency_level",
            "text_request_type",
            "text_request_reason",
            "subject_body_divider",
            "include_link",
            "text_request_length",
        ]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
