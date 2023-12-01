from openai_functions import response_gen
from prompts import PROJECT_UNIQUENESS_PROMPT


class Unique:
    def __init__(self, description, readme):
        self.description = description
        self.readme = readme

    def _evaluate_code_unique(self):
        unqiueness_report = response_gen(self.description + "\n" + self.readme, PROJECT_UNIQUENESS_PROMPT, 0)
        return unqiueness_report
