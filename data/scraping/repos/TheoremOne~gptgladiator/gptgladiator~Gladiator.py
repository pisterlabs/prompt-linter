import concurrent.futures
import json
import ast
from gptgladiator import prompts
from gptgladiator import mocks
from typing import Any, Iterator
import openai as openai
from gptgladiator.GPTModel import GptModel
from gptgladiator.ChatBot import ChatBot


class Gladiator():
    mock_responses = False
    mock_grades = False
    debug_output = False

    def __init__(self, api_key, num_drafts=3):
        """
        Initialize with an OpenAI `api_key` and the number of drafts to generate.

        By default, `gpt-3.5-turbo` will be used to generate the drafts and
        `gpt-4` will be used to grade them. You can change these models by
        setting the `generate_model` and `grade_model` attributes after setup.

        To test the gladiator without using OpenAI, set the `mock_responses`
        and `mock_grades` attributes to `True`. This will cause the gladiator
        to use the data in the `mocks` module instead of using OpenAI's API.

        To see extra debug output (including underlying errors if they occur)
        printed to the console, set the `debug_output` attribute to `True`.
        """
        openai.api_key = api_key
        self.generate_model = GptModel('gpt-3.5-turbo', 4000)
        self.grade_model = GptModel('gpt-4', 8000)
        self.drafts = []
        self.grades = []
        self.num_drafts = num_drafts

    def run(self, prompt: str):
        """
        Run the gladiator with a given `prompt` and return the winning response.

        All of the drafts that were considered and their respective grades are
        available via the `drafts` and `grades` methods after running.
        """
        drafts = self.generate_drafts(prompt)
        grades_in_json = self.grade_drafts(drafts)
        winning_index, winning_content = self.select_winner(
            drafts, grades_in_json)
        return winning_content

    def drafts(self):
        """
        Returns the drafts that were generated by the `generate_model`.
        """
        return self.drafts

    def grades(self):
        """
        Returns the grades that were generated by the `grade_model`.
        These correspond in order to the `drafts`.
        """
        return self.grades

    def generate_drafts(self, prompt):
        """
        Generate drafts using the `generate_model`, store them, and return them.

        This is part of what the `run` method does, but you can call this
        method directly if you want to see the drafts before grading them.
        """
        if self.debug_output:
            print(f"running: {self.num_drafts} times with prompt: {prompt}")

        prompts = [prompt] * self.num_drafts
        drafts = self._concurrent_requests(
            prompts) if not self.mock_responses else mocks.mock_responses

        self.drafts = drafts
        return drafts

    def grade_drafts(self, drafts):
        """
        Grade the given `drafts` using the `grade_model` and return the grades.

        This is part of what the `run` method does, but you can call this
        method directly if you want to see the grades before selecting a winner,
        or if you want to supply the drafts yourself.
        """
        gradingbot = ChatBot(self.grade_model, messages=[],
                             debug_output=self.debug_output)
        response = gradingbot.get_completion(prompts.make_grading_prompt(
            drafts)) if not self.mock_grades else mocks.mock_grades

        if self.debug_output:
            print("response to parse to json = ", response)

        grades_json = parse_json(response)
        self.grades = grades_json

        if self.debug_output:
            print("grades = ", grades_json)

        return grades_json

    def select_winner(self, drafts, grades_json):
        """
        Select the winning draft from the given `drafts` and `grades_json`.

        This is part of what the `run` method does, but you can call this
        method directly if you want to supply the drafts and grades yourself.
        """
        winning_index = max(range(len(grades_json)), key=lambda i: int(
            grades_json[str(i + 1)]['score']))

        if self.debug_output:
            print("winning_index = ", winning_index)

        winning_content = drafts[winning_index]
        return winning_index, winning_content

    def _process(self, tuple):
        """
        Process a single prompt and return the response.
        """
        i, prompt = tuple
        temperature = 1-i*.1

        if self.debug_output:
            print("temperature = ", temperature)

        chatbot = ChatBot(self.generate_model, temperature=temperature,
                          messages=[], debug_output=self.debug_output)
        response = chatbot.get_completion(prompt)
        return response

    def _concurrent_requests(self, prompts, max_active_tasks=10) -> Iterator[Any]:
        """
        Run the `_process` method concurrently on the given `prompts`.
        """
        max_active_tasks = len(prompts) if len(
            prompts) < max_active_tasks else max_active_tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_active_tasks) as executor:
            results = executor.map(self._process, enumerate(prompts))
        return list(results)


def parse_json(json_response: str) -> dict:
    try:
        return json.loads(json_response)
    except Exception as e:
        print(f'Error parsing the json response. \n'
              f'{e}\n'
              f'Defaulting to literal eval.')
        return ast.literal_eval(json_response)