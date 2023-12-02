import unittest
import os

import cohere

from weblm.crawler import Crawler
from weblm.controller import Controller
from weblm.parser import Parser, TasksInterface


class TestParser(unittest.TestCase):
    def setUp(self) -> None:

        # find a better webpage, maybe news or something from a common benchmark?
        self.webpage = "https://txt.cohere.ai/free-developer-tier-announcement/"
        self.crawler = Crawler(headless=True)
        co = cohere.Client(os.environ.get("COHERE_KEY"), check_api_key=False)
        self.controller = Controller(co, "Generate a summary of this webpage")

    def test_summary(self):

        self.crawler.go_to_page(self.webpage)
        content = self.crawler.page.content()
        parser = Parser(content)
        readable_text = parser.process()

        self.assertIsNotNone(readable_text)

        task_interface = TasksInterface()
        short_text_with_prompt = task_interface.summary(text=readable_text)
        summary_response = self.controller.use_text(prompt=short_text_with_prompt)
        self.assertIsNotNone(summary_response)
