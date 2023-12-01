import datetime
import json
import os
import random
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import openai
import tiktoken

import chat_wrapper as cw
from settings import API_KEY
from templates import GetTemplates, template_selector


class NoTemplateSelectedError(Exception):
    def __init__(self, message: str = None):
        if message is None:
            message = "No template selected"
        super().__init__(message)


class ChatLogAndGPTChatFactory:
    """T
    his class is used to make ChatLog and GPTChat objects from templates
    Relies on:
        - template_selector: GetTemplates can use custom instance of this class (Separate instances might be working with a different set of templates, but functionality is the same)
        - GPTChat: GPTChat class from GPTchat.py
        - ChatLog: ChatLog class from ChatHistory.py
    Attributes:
        - API_KEY: OpenAI API key, required to make GPTChat objects
        - template_selector: GetTemplates object, used to get templates
        - selected_template: dict, selected template, selected in the select_template method

    Methods:
        - select_template(template_name: str): selects a template from the template_selector, by name
        - _make_chat_log: makes a ChatLog object from a template
        - _make_gpt_chat: makes a GPTChat object from a template
        - make_chat_log_and_gpt_chat: makes a ChatLog and GPTChat object from a template, returns a tuple of them
    Example Usage:
        template_name = template_selector.search_by_tag("gpt-4")[0]
        factory = ChatLogAndGPTChatFactory(API_KEY)
        factory.select_template(template_name)
        chat_log, gpt_chat = factory.make_chat_log_and_gpt_chat()

    """

    def __init__(self, API_KEY, template_selector: GetTemplates = template_selector):
        self.API_KEY = API_KEY
        self.template_selector = template_selector
        self.selected_template = template_selector.get_template("gpt-4_default")

    def select_template(self, template_name: str) -> None:
        self.selected_template = self.template_selector.get_template(template_name)

    def _make_chat_log(self, template: dict) -> cw.g.ch.ChatLog:
        """Makes a ChatLog object from a template"""
        settings: dict = template["chat_log"]

        chat_log = cw.g.ch.ChatLog(**settings)
        return chat_log

    def _make_gpt_chat(self, template: dict) -> cw.g.GPTChat:
        """Makes a GPTChat object from a template"""
        settings: dict = template["gpt_chat"]

        gpt_chat = cw.g.GPTChat(API_KEY=self.API_KEY, template=template, **settings)
        return gpt_chat

    def make_chat_log_and_gpt_chat(self) -> tuple[cw.g.ch.ChatLog, cw.g.GPTChat]:
        """Makes a ChatLog and GPTChat object from a template, returns a tuple of them"""
        if self.selected_template is None:
            raise NoTemplateSelectedError()
        chat_log = self._make_chat_log(self.selected_template)
        gpt_chat = self._make_gpt_chat(self.selected_template)
        return (chat_log, gpt_chat)


class TestChatLogAndGPTChatFactory(unittest.TestCase):
    def setUp(self):
        self.factory = ChatLogAndGPTChatFactory(API_KEY)
        self.template_selector = template_selector
        self.test_template = template_selector.get_template("gpt-4_default")
        self.test_template_name = "gpt-4_default"

    def test_select_template(self):
        """Tests that the select_template method works"""
        template_name = self.factory.template_selector.search_templates_by_tag("gpt-4")[
            0
        ]
        self.factory.select_template(template_name)
        self.assertIsNotNone(self.factory.selected_template)

    def test_make_chat_log(self):
        """Tests that the _make_chat_log method works"""
        template_name = self.test_template_name
        self.factory.select_template(template_name)
        chat_log = self.factory._make_chat_log(self.test_template)
        self.assertIsInstance(chat_log, cw.g.ch.ChatLog)

    def test_make_gpt_chat(self):
        """Tests that the _make_gpt_chat method works"""
        template_name = self.test_template_name
        self.factory.select_template(template_name)
        gpt_chat = self.factory._make_gpt_chat(self.factory.selected_template)
        self.assertIsInstance(gpt_chat, cw.g.GPTChat)

    def test_make_chat_log_and_gpt_chat(self):
        """Tests that the make_chat_log_and_gpt_chat method works"""
        self.factory.select_template(self.test_template_name)
        chat_log, gpt_chat = self.factory.make_chat_log_and_gpt_chat()
        self.assertIsInstance(gpt_chat, cw.g.GPTChat)
        self.assertIsInstance(chat_log, cw.g.ch.ChatLog)

    def tearDown(self):
        del self.factory


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)


class ChatWrapperFactory:
    """A factory class for making ChatWrapper objects from a template
    Methods:
        select_template(template_name: str): selects a template from the template_selector, by name
        search_templates_by_tag(tag: str, return_type = 'name'): searches for templates by tag, returns a list of template names or templates
        make_chat_wrapper(template: dict): makes a ChatWrapper object from a template
    """

    def __init__(
        self,
        API_KEY=API_KEY,
        template_selector: GetTemplates = template_selector,
    ) -> None:
        self.template_selector = template_selector
        self.selected_template = self.template_selector.get_template("gpt-4_default")
        self.chat_and_gpt_factory = ChatLogAndGPTChatFactory(API_KEY, template_selector)
        self.api_key = API_KEY

    def select_template(self, template_name: str) -> None:
        """Selects a template from the template_selector, by name, will raise a TemplateNotFoundError if the template is not found"""
        self.selected_template = self.template_selector.get_template(template_name)
        self.chat_and_gpt_factory.select_template(template_name)

    def search_templates_by_tag(
        self, tag: str, return_type="name"
    ) -> List[str] | List[dict]:
        """Searches for templates by tag and returns a list of them, either as the names(keys) or the templates themselves(values). If none are found, returns an empty list"""
        return self.template_selector.search_templates_by_tag(tag)

    def make_chat_log_and_gpt_chat(self) -> tuple[cw.g.ch.ChatLog, cw.g.GPTChat]:
        """Using the selected template, makes a ChatLog and GPTChat object from a template, returns a tuple of them"""
        return self.chat_and_gpt_factory.make_chat_log_and_gpt_chat()

    def select_default_for_model(self, model: str) -> None:
        try:
            self.select_template(f"{model}_default")
        except template_selector.TemplateNotFoundError:
            print(f"Template {model}_default not found, selecting gpt-4_default")
            self.select_template("gpt-4_default")

    def make_chat_wrapper(self, template_name: str = None) -> cw.ChatWrapper:
        """Makes a ChatWrapper object from a template and returns it"""
        if template_name is not None:
            self.select_template(template_name)
        if template_name is None and self.selected_template is None:
            raise NoTemplateSelectedError()
        chat_log, gpt_chat = self.make_chat_log_and_gpt_chat()
        chat_wrapper = cw.ChatWrapper(
            API_KEY=self.api_key, chat_log=chat_log, gpt_chat=gpt_chat
        )

        return chat_wrapper


wrapper_factory = ChatWrapperFactory(API_KEY, template_selector)
