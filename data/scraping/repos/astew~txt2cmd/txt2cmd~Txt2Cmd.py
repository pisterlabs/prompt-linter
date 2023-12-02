import logging

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

from config import settings
from txt2cmd.prompt import chat_template, template
from txt2cmd.prompt.output_parser import CodeOutputParser


class Txt2Cmd:
    def __init__(self) -> None:
        self.llm = OpenAI(
            model=settings.OPENAI_TXT_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
        )

    def generate_script(self, user_prompt: str, language: str, content: str = None) -> str:
        """Templates user prompt for LLM and parses script from response using prediction model

        Args:
            user_prompt (str): User defined prompt.
            language (str): Programming language of script.
            content (str, optional): Content of existing script if updating. Defaults to None.

        Returns:
            str: Generated code
        """
        prompt = (
            template.update_script.format(user_prompt=user_prompt, language=language, content=content)
            if content
            else template.new_script.format(user_prompt=user_prompt, language=language)
        )
        response = self.llm.predict(prompt)
        return response.replace("```", "")


class ChatTxt2Cmd:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
        )
        self.parser = CodeOutputParser()

    def generate_script(self, user_prompt: str, language: str, content: str = None) -> str:
        """Templates user prompt for LLM and parses script from response using Chat completion model

        Args:
            user_prompt (str): User defined prompt.
            language (str): Programming language of script.
            content (str, optional): Content of existing script if updating. Defaults to None.

        Returns:
            str: Generated code
        """
        prompt = (
            chat_template.update_script.format_prompt(
                action="updating", language=language, content=content, user_prompt=user_prompt
            ).to_messages()
            if content
            else chat_template.new_script.format_prompt(
                action="creating", language=language, user_prompt=user_prompt
            ).to_messages()
        )

        logging.debug("\n".join([x.content for x in prompt]))

        response = self.llm.predict_messages(prompt).content
        code_str = self.parser.parse(response)
        return code_str
