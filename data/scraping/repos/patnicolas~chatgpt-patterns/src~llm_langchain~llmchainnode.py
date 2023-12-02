__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2022, 23. All rights reserved."

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from collections.abc import Callable
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import openai
from typing import Dict, AnyStr, Any, List


class LLMChainNode(object):

    def __init__(self, temp: float, template: Callable[Dict[AnyStr, Any], AnyStr]):
        self.template = template
        self.chat = ChatOpenAI(temperature=temp)

    def get_arguments(self, **kwargs: Dict[AnyStr, Any]) -> List[AnyStr]:
        prompt_template = self.__get_prompt(**kwargs)
        return prompt_template.messages[0].input_variables

    def __call__(self, **kwargs: Dict[AnyStr, Any]) -> AnyStr:
        """
        Generic call for the
        @param kwargs: Dictionary arguments
        @return: Response from LLM
        """
        import logging
        try:
            prompt_template = self.__get_prompt(**kwargs)
            new_messages = prompt_template.format_messages(**kwargs)

            answer = self.chat(new_messages)
            return answer.content
        except openai.error.AuthenticationError as e:
            logging.error(str(e))
            return ""

    def get_output_parser(self, input_message: AnyStr, output_format: Dict[AnyStr, AnyStr]) -> Dict[AnyStr, AnyStr]:
        acc = []
        response_schemas = []
        for param_name, param_desc in output_format.items():
            acc.append(f'{param_name}: ${param_desc}')
            response_schemas.append(ResponseSchema(name=param_name, description=param_desc))

        params = '\n'.join(acc)
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        appended_params = f'Format the output as JSON with the following keys\n{params}'
        updated_content = f'{input_message}\n{appended_params}'
        template_with_output_params = f'{updated_content}\n{format_instructions}'
        prompt_template = ChatPromptTemplate.from_template(template_with_output_params)
        message_with_formatted_output = prompt_template.format_messages(
            text=updated_content,
            format_instructions=format_instructions
        )
        answer = self.chat(message_with_formatted_output)
        return output_parser.parse(answer.content)

    def __get_prompt(self, **kwargs):
        template_str = self.template(**kwargs)
        return ChatPromptTemplate.from_template(template_str)


def get_template(**kwargs: Dict[AnyStr, Any]) -> AnyStr:
    language = kwargs['language']
    text = kwargs['text']
    instruction = f'Translate the text that is delimited by triple backticks into a language {language}. text ```{text}```'
    return instruction


if __name__ == '__main__':
    chat_gpt_chain = LLMChainNode(0.2, get_template)
    response = chat_gpt_chain(**{'language': 'french', 'text': 'this is a good time to walk'})
    print(response)
