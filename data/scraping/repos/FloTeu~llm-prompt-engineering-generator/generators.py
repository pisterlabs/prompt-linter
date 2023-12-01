import logging
import json
from typing import Optional, Type, List

from langchain.chains import LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import OutputParserException
from pydantic import BaseModel

from llm_prompting_gen.models.prompt_engineering import PromptEngineeringMessages, PromptElements


class PromptEngineeringGenerator:
    """
    Combines Prompt Engineering dataclass with LLM.
    """

    def __init__(self, llm: BaseLanguageModel, prompt_elements: Optional[PromptElements] = None,
                 message_order: List[str] = None):
        """

        :param llm: Large Language Model, which executed the task
        :param prompt_elements: All prompt elements which contain prompt engineering techniques
        :param message_order: List of field PromptElement field names. Defines final order of prompt elements. Also, additional prompt element fields can be included. examples=["role", "instruction", "input", "output"]
        """
        self.llm = llm
        self.prompt_elements: PromptElements = prompt_elements or PromptElements()
        self.message_order: List[str] = message_order

    @classmethod
    def from_json(cls, file_path: str, llm: BaseLanguageModel):
        with open(file_path, "r") as fp:
            data_dict = json.load(fp)
        prompt_elements = PromptElements(**data_dict)
        return cls(llm=llm, prompt_elements=prompt_elements, message_order=list(data_dict.keys()))

    def _get_messages(self) -> PromptEngineeringMessages:
        """Transform the prompt elements to langchain message dataclass"""
        return PromptEngineeringMessages.from_pydantic(self.prompt_elements)

    def _get_llm_chain(self) -> LLMChain:
        """Combines chat messages with LLM and returns a LLM Chain"""
        chat_prompt = self.get_chat_prompt()
        return LLMChain(llm=self.llm, prompt=chat_prompt)

    def get_chat_prompt(self) -> ChatPromptTemplate:
        """Combines prompt elements to chat messages"""
        chat_prompt: ChatPromptTemplate = self._get_messages().get_chat_prompt_template(
            message_order=self.message_order)
        return chat_prompt

    def get_final_prompt(self, **kwargs) -> str:
        """Returns the final prompt including all prompt elements as string"""
        chat_prompt = self.get_chat_prompt()
        return chat_prompt.format(**kwargs)

    def generate(self, *args, **kwargs) -> str:
        """Generates a llm str output based on prompt engineering elements"""
        assert self.prompt_elements.is_any_set()
        llm_chain = self._get_llm_chain()
        if bool(args):
            return llm_chain.run(*args, **kwargs)
        else:
            # predict() can be called without any arguments provided
            return llm_chain.predict(**kwargs)


class ParsablePromptEngineeringGenerator(PromptEngineeringGenerator):
    """
    Enhances PromptEngineeringGenerator with pydantic output format.
    LLM output will be parsed to pydantic.
    Prompt element output format is ignored, if pydantic is provided.
    """

    def __init__(self, llm: BaseLanguageModel, pydantic_cls: Type[BaseModel],
                 prompt_elements: Optional[PromptElements] = None, message_order: List[str] = None):
        super().__init__(llm=llm, prompt_elements=prompt_elements, message_order=message_order)
        # Set up a parser
        self.output_parser = PydanticOutputParser(pydantic_object=pydantic_cls)
        self.prompt_elements.output_format = self.output_parser.get_format_instructions()
        # If output_format was not included, append it to end of the message order
        if "output_format" not in self.message_order:
            self.message_order.append("output_format")

    @classmethod
    def from_json(cls, file_path: str, llm: BaseLanguageModel, pydantic_cls: Type[BaseModel]):
        with open(file_path, "r") as fp:
            data_dict = json.load(fp)
        prompt_elements = PromptElements(**data_dict)
        return cls(llm=llm, pydantic_cls=pydantic_cls, prompt_elements=prompt_elements,
                   message_order=list(data_dict.keys()))

    def generate(self, *args, **kwargs) -> BaseModel:
        """Generates a pydantic parsed object"""
        llm_output = super().generate(*args, **kwargs)
        try:
            # return parsed output
            return self.output_parser.parse(llm_output)
        except OutputParserException:
            logging.warning("Could not parse llm output to pydantic class. Retry...")
            # Retry to get right format
            retry_parser = RetryWithErrorOutputParser.from_llm(parser=self.output_parser, llm=self.llm)
            _input = self._get_llm_chain().prompt.format_prompt(**kwargs)
            return retry_parser.parse_with_prompt(llm_output, _input)
