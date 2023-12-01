import logging
import openai
import os
from pathlib import Path


# directory path
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# logging
logger = logging.getLogger(__name__)


class Solution:
    def __init__(self, model, system_msg, temperature, prompt_file, use=False):
        self.model = model
        self.system_msg = system_msg
        self.temperature = temperature
        self.prompt_path = Path(prompt_file).resolve()
        self.use = use

    def generate_solution(self, disease):
        if not self.use:
            return f"Using more water helps with {disease}"

        prompt = self.generate_solution_prompt(disease)
        try:
            logger.info(f"Generating solution for disease {disease}")
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error(str(e))
            raise ValueError(str(e))

        return response.choices[0]["message"]["content"]



    def generate_solution_prompt(self, disease):
        with open(self.prompt_path, "r") as f:
            template = f.read()
        print(template)
        return template + f" The disease of the plant is {disease}."
